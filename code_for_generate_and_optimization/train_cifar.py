import argparse
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import script_utils
from torch.optim.lr_scheduler import StepLR
from utils import *
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, data_folder, sequences):
        self.data = data_folder
        self.sequences = sequences
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sequence = self.sequences[idx]
        return torch.tensor(sample, dtype=torch.float32), sequence


def get_sequence_list_from_csv(file_path):
    df = pd.read_csv(file_path, usecols=["switch", "stem1", "stem2"])
    
    sequence_list = df.apply(lambda row: row['switch'] + row['stem1'] + row['stem2'], axis=1).tolist()
    
    return sequence_list


def main():
    loss_flag = 0.15
    test_loss_save = []
    epoch_history = []
    train_loss_history = []
    train_diffusion_loss_history = []
    train_embedding_loss_history = []
    test_loss_history = []
    test_diffusion_loss_history = []
    test_embedding_loss_history = []
    args = create_argparser().parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        print("=" * 60)
        print("Verifying forward pass consistency...")
        diffusion.eval()
        
        with torch.no_grad():
            x_test = torch.randn(4, 1, 4, 44).to(device)
            t_test = torch.randint(0, 1000, (4,)).to(device)
            
            out1 = diffusion.model(x_test, t_test, return_features=False)
            
            out2 = diffusion.ema_model(x_test, t_test)
            
            diff = (out1 - out2).abs().max().item()
            print(f"Output difference: {diff:.2e}")
            
            if diff > 1e-5:
                print("CRITICAL ERROR: Forward pass inconsistent!")
                print("   Trained model cannot be used for sampling")
                print("   Please check modifications in unet.py and diffusion.py")
                return
            else:
                print("PASSED: Forward pass is consistent")
        
        print("=" * 60)
        diffusion.train()
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))

        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        batch_size = args.batch_size

        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

        all_sample = get_sequence_list_from_csv(
            file_path='../data/Toehold_mRNA_Dataset_cleanplus.csv'
        )
        train_size = int(0.8 * len(all_sample))

        
        train_sequences = all_sample[:train_size]
        test_sequences = all_sample[train_size:]
        
        encoded_sequence_train = []
        for sequence in train_sequences:
            if len(sequence) != 45:
                print('error!!!')
            encoded_sequence = one_hot_encoding(sequence[1:])
            encoded_sequence_train.append(encoded_sequence)
            
        encoded_sequence_test = []
        for sequence in test_sequences:
            if len(sequence) != 45:
                print('error!!!')
            encoded_sequence = one_hot_encoding(sequence[1:])
            encoded_sequence_test.append(encoded_sequence)

        train_array = np.array(encoded_sequence_train)
        test_array = np.array(encoded_sequence_test)

        print('train_array.shape =', train_array.shape)
        print('test_array.shape =', test_array.shape)

        train_array = np.expand_dims(train_array, axis=1)
        test_array = np.expand_dims(test_array, axis=1)

        print("train_array shape after expansion:", train_array.shape)
        print("test_array shape after expansion:", test_array.shape)
        
        train_dataset = CustomDataset(train_array, train_sequences)
        test_dataset = CustomDataset(test_array, test_sequences)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if args.use_large_model:
            dummy_x = torch.randn(2, 1, 4, 44).to(device)
            dummy_t = torch.randint(0, 1000, (2,)).to(device)
            output, features = diffusion.model(dummy_x, dummy_t, return_features=True)
            print(f"UNet output shape: {output.shape}")
            print(f"Bottleneck feature shape: {features.shape}")

        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):
            diffusion.train()
            acc_train_loss = 0
            acc_train_diffusion_loss = 0
            acc_train_embedding_loss = 0
            for x, sequences in train_loader:
                x = x.to(device)
                loss_dict = diffusion(x, y=None, sequences=sequences)
                total_loss = loss_dict['total_loss']
                acc_train_loss += total_loss.item()
                acc_train_diffusion_loss += loss_dict['diffusion_loss'].item()
                acc_train_embedding_loss += loss_dict['embedding_loss'].item()


                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                diffusion.update_ema()

            avg_train_loss = acc_train_loss / len(train_loader)
            avg_train_diffusion_loss = acc_train_diffusion_loss / len(train_loader)
            avg_train_embedding_loss = acc_train_embedding_loss / len(train_loader)
            print(f'Epoch {iteration}, Train Loss = {avg_train_loss:.4f}')
            scheduler.step()

            test_loss = 0
            test_diffusion_loss = 0
            test_embedding_loss = 0
            with torch.no_grad():
                diffusion.eval()
                for x, sequences in test_loader:
                    x = x.to(device)
                    loss_dict = diffusion(x, y=None, sequences=sequences,force_embedding=True)
                    test_loss += loss_dict['total_loss'].item()
                    test_diffusion_loss += loss_dict['diffusion_loss'].item()
                    test_embedding_loss += loss_dict['embedding_loss'].item()
            
            avg_test_loss = test_loss / len(test_loader)
            avg_test_diffusion_loss = test_diffusion_loss / len(test_loader)
            avg_test_embedding_loss = test_embedding_loss / len(test_loader)
            print(f'Epoch {iteration}, Test Loss = {avg_test_loss:.4f} '
                        f'(Diffusion: {avg_test_diffusion_loss:.4f}, '
                        f'Embedding: {avg_test_embedding_loss:.4f})')
            
            epoch_history.append(iteration)
            train_loss_history.append(avg_train_loss)
            train_diffusion_loss_history.append(avg_train_diffusion_loss)
            train_embedding_loss_history.append(avg_train_embedding_loss)
            test_loss_history.append(avg_test_loss)
            test_diffusion_loss_history.append(avg_test_diffusion_loss)
            test_embedding_loss_history.append(avg_test_embedding_loss)
            
            loss_df = pd.DataFrame({
                'epoch': epoch_history,
                'train_total_loss': train_loss_history,
                'train_diffusion_loss': train_diffusion_loss_history,
                'train_embedding_loss': train_embedding_loss_history,
                'test_total_loss': test_loss_history,
                'test_diffusion_loss': test_diffusion_loss_history,
                'test_embedding_loss': test_embedding_loss_history
            })
            csv_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-loss_history.csv"
            loss_df.to_csv(csv_filename, index=False)
                
            if iteration % args.log_rate == 0:


                
                diffusion.eval()
                sequences=[]
                samples = diffusion.sample(2048, device)
                samples = samples.squeeze(dim=1)
                print('samples.shape = ', samples.shape)

                samples = samples.to('cpu').detach().numpy()


                for j in range(samples.shape[0]):

                    decoded_sequence = decode_one_hot(samples[j])
                    sequences.append("A" + decoded_sequence)


                k_mer_fre_cor = calculate_overall_kmer_correlation(dataset1=sequences, dataset2=all_sample, k=6)
                print(f'epoch = {iteration}, 6_mer_fre_cor = {k_mer_fre_cor}')
                make_fasta_file(sequences,path=f'../sequence/embweight_0.1(3)_Switche_epoch={iteration}_6_mer_fre_cor={k_mer_fre_cor}.fasta')
            
            test_loss_save.append(avg_test_loss)
            if avg_test_loss < loss_flag:
                loss_flag = avg_test_loss
                print('Saving best model')
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-best-model.pth"
                torch.save(diffusion.state_dict(), model_filename)

            if iteration % args.checkpoint_rate == 0:
                test_loss_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-test_loss.npy"
                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-model.pth"
                optim_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}-optim.pth"

                np.save(test_loss_filename, np.array(test_loss_save))
                torch.save(diffusion.state_dict(), model_filename)
                torch.save(optimizer.state_dict(), optim_filename)
                print(f"Checkpoint saved to: {args.log_dir}")

            acc_train_loss = 0


        print(f"Training complete! Loss history saved to: {csv_filename}")

    except KeyboardInterrupt:
        print("Keyboard interrupt, run finished early")
        if len(epoch_history) > 0:
            loss_df = pd.DataFrame({
                'epoch': epoch_history,
                'train_loss': train_loss_history,
                'test_loss': test_loss_history
            })
            csv_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-loss_history.csv"
            loss_df.to_csv(csv_filename, index=False)
            print(f"Loss history saved to: {csv_filename}")


def create_argparser():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")

    defaults = dict(
        learning_rate=1e-4,
        batch_size=1024,
        iterations=500,

        log_to_wandb=False,
        log_rate=50,
        checkpoint_rate=50,
        log_dir="../result",
        project_name='Switch',
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,

        device=device,
        
    )

    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
