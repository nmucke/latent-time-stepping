



import pdb
from matplotlib import pyplot as plt
import torch.nn as nn
import torch
from tqdm import tqdm
import yaml
from latent_time_stepping.AE_models.autoencoder import Autoencoder
from latent_time_stepping.AE_models.encoder_decoder import Decoder, Encoder
from latent_time_stepping.datasets.AE_dataset import AEDataset

LOCAL_LOAD_PATH = f'../../../../../scratch2/ntm/data/multi_phase/raw_data/train'


class Autoencoder_(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16, bias=False),
        )

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1024),
            nn.LeakyReLU(),
            nn.Unflatten(dim=1, unflattened_size=(128, 8)),
            nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=8, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose1d(in_channels=4, out_channels=2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.LeakyReLU(),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        
        x_hat = self.decoder(z)
        return x_hat

def main():

    dataset = AEDataset(
        local_path=LOCAL_LOAD_PATH,
        sample_ids=range(0, 1),
        preprocessor=None,
        num_skip_steps=5,
        end_time_index=500,
        filter=True,
        states_to_include=(1,2),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    #model = Autoencoder()


    config_path = f"configs/neural_networks/multi_phase_WAE.yml"
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    encoder = Encoder(**config['model_args']['encoder'])
    decoder = Decoder(**config['model_args']['decoder'])

    model = Autoencoder(
        encoder=encoder,
        decoder=decoder,
    )
    model = model.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    loss_fn = torch.nn.MSELoss()

    model.train()

    state, pars = next(iter(dataloader))
    state = state[:, :, :, 20:21]
    state = state.to('cuda')
    pars = pars.to('cuda')

    pbar = tqdm(
            range(500),#enumerate(dataloader),
            total=int(len(dataloader.dataset)/dataloader.batch_size),
            bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'
        )
    for epoch in range(20000):
        
        total_loss = 0
        
        optimizer.zero_grad()

        z = model.encoder(state)
        out = model.decoder(z, pars)

        loss_val = loss_fn(state, out)

        loss_val.backward()
        optimizer.step()


        pbar.set_postfix({'loss': loss_val.item(), 'epoch': epoch}) 

        '''
        batch_size = state.shape[0]
        num_time_steps = state.shape[-1]

        #state = state.permute(0, 3, 1, 2)
        #state = state.reshape(batch_size*num_time_steps, 2, 512)

        state = state[:, :, :, 20]

        out = model(state)

        loss_val = loss_fn(state, out)

        loss_val.backward()
        optimizer.step()

        total_loss += loss_val.item()

        if i % 10 == 0:
            pbar.set_postfix({'loss': total_loss/(i+1), 'epoch': epoch}) 
        '''

    model.eval()

    model.to('cpu')

    state = state.to('cpu')
    pars = pars.to('cpu')
    #state = state.permute(0, 3, 1, 2)
    #state = state.reshape(batch_size*num_time_steps, 2, 512)

    z = model.encoder(state)
    out = model.decoder(z, pars)

    plt.figure()
    plt.plot(state[0, 1, :].detach().numpy())
    plt.plot(out[0, 1, :].detach().numpy())
    plt.plot(state[0, 0, :].detach().numpy())
    plt.plot(out[0, 0, :].detach().numpy())
    plt.show()


    return 0


if __name__ == "__main__":
    main()