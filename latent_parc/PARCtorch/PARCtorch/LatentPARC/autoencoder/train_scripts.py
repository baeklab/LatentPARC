import torch
import numpy as np
from tqdm import tqdm
from utils import save_model_and_logs

def train_autoencoder(model, optimizer, loss_function, train_loader, val_loader, 
                      device, epochs=3000, save_every_epochs = 100, image_size=(64, 64), n_channels=3, 
                      scheduler=None, noise_fn=None, initial_max_noise=0.16, 
                      n_reduce_factor=0.5, reduce_on=1000, save_path=None, weights_name=None):
    """ Train an autoencoder with optional noise injection. """

    log_dict = {'training_loss_per_epoch': [], 'validation_loss_per_epoch': []}
    
    model.to(device)

    max_noise = initial_max_noise  # Initial noise level

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # Reduce noise periodically
        if (epoch + 1) % reduce_on == 0:
            max_noise *= n_reduce_factor

        # --- Training ---
        model.train()
        train_losses = []
        for images in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            images = images[0][:, 0:n_channels, ...].to(device)
            
            # Apply noise if function is provided
            noisy_images = noise_fn(images, max_val=max_noise) if noise_fn else images

            # Forward pass
            output = model(noisy_images)
            loss = loss_function(output, images.view(-1, n_channels, *image_size))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        log_dict['training_loss_per_epoch'].append(avg_train_loss)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for val_images in tqdm(val_loader, desc="Validating"):
                val_images = val_images[0][:, 0:n_channels, ...].to(device)

                # Forward pass
                output = model(val_images)
                val_loss = loss_function(output, val_images.view(-1, n_channels, *image_size))
                val_losses.append(val_loss.item())

        avg_val_loss = np.mean(val_losses)
        log_dict['validation_loss_per_epoch'].append(avg_val_loss)

        print(f"Epoch {epoch+1}: Training Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

        if scheduler:
            scheduler.step(avg_val_loss)

        if (epoch + 1) % save_every_epochs == 0:
            save_model_and_logs(
                model=model,
                optimizer=optimizer,
                save_path=save_path,
                weights_name=weights_name,
                log_dict=log_dict,
                epoch=epoch,
                scheduler=scheduler,
                current_max_noise=max_noise,
                
                config={ # just for settings info, not needed for commencing training
                    "initial_lr": optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"]),
                    "current_lr": optimizer.param_groups[0]["lr"],
                    "image_size": image_size,
                    "n_channels": n_channels,
                },
            
                noise_config={ # just for settings info, not needed for commencing training
                    "initial_max_noise": initial_max_noise,
                    "n_reduce_factor": n_reduce_factor,
                    "reduce_on": reduce_on,
                },
            )

    # Save model and logs
    
    save_model_and_logs(
                model=model,
                optimizer=optimizer,
                save_path=save_path,
                weights_name=weights_name,
                log_dict=log_dict,
                epoch=epoch,
                scheduler=scheduler,
                current_max_noise=max_noise,
            
                config={
                    "initial_lr": optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"]),
                    "current_lr": optimizer.param_groups[0]["lr"],
                    "image_size": image_size,
                    "n_channels": n_channels,
                },
            
                noise_config={
                    "initial_max_noise": initial_max_noise,
                    "n_reduce_factor": n_reduce_factor,
                    "reduce_on": reduce_on,
                },
            )
    
    return log_dict

## TODO: Finish modifying below to work with BasicAutoencoder in backbones, then replace above with this
# def train_basic_autoencoder(model, optimizer, loss_function, train_loader, val_loader,
#                       device, epochs=3000, save_every_epochs=100, image_size=(64, 64), n_channels=3,
#                       scheduler=None, noise_fn=None, initial_max_noise=0.16,
#                       n_reduce_factor=0.5, reduce_on=1000, save_path=None, weights_name=None
#                       ):
#     """ Train a basic autoencoder. Model returns (log dict). """

#     log_dict = {
#         'training_loss_per_epoch': [],
#         'validation_loss_per_epoch': [],
#         # Extra: track the two components so you can see which is driving training
#         'training_sharp_loss_per_epoch': [],
#         'training_blurry_loss_per_epoch': [],
#         'validation_sharp_loss_per_epoch': [],
#         'validation_blurry_loss_per_epoch': [],
#     }

#     model.to(device)
#     max_noise = initial_max_noise

#     for epoch in range(epochs):
#         print(f"\nEpoch {epoch + 1}/{epochs}")

#         if (epoch + 1) % reduce_on == 0:
#             max_noise *= n_reduce_factor

#         # --- Training ---
#         model.train()
#         train_losses, train_sharp_losses, train_blurry_losses = [], [], []

#         for images in tqdm(train_loader, desc="Training"):
#             optimizer.zero_grad()
#             images = images[0][:, 0:n_channels, ...].to(device)
#             noisy_images = noise_fn(images, max_val=max_noise) if noise_fn else images

#             targets = images.view(-1, n_channels, *image_size)

#             # Forward pass — now returns two outputs
#             sharp, blurry = model(noisy_images)

#             loss_sharp  = loss_function(sharp,  targets)
#             loss_blurry = loss_function(blurry, targets)
#             loss = sharp_weight * loss_sharp + blurry_weight * loss_blurry

#             loss.backward()
#             optimizer.step()

#             train_losses.append(loss.item())
#             train_sharp_losses.append(loss_sharp.item())
#             train_blurry_losses.append(loss_blurry.item())

#         avg_train_loss        = np.mean(train_losses)
#         avg_train_sharp_loss  = np.mean(train_sharp_losses)
#         avg_train_blurry_loss = np.mean(train_blurry_losses)

#         log_dict['training_loss_per_epoch'].append(avg_train_loss)
#         log_dict['training_sharp_loss_per_epoch'].append(avg_train_sharp_loss)
#         log_dict['training_blurry_loss_per_epoch'].append(avg_train_blurry_loss)

#         # --- Validation ---
#         model.eval()
#         val_losses, val_sharp_losses, val_blurry_losses = [], [], []

#         with torch.no_grad():
#             for val_images in tqdm(val_loader, desc="Validating"):
#                 val_images = val_images[0][:, 0:n_channels, ...].to(device)
#                 targets = val_images.view(-1, n_channels, *image_size)

#                 sharp, blurry = model(val_images)

#                 loss_sharp  = loss_function(sharp,  targets)
#                 loss_blurry = loss_function(blurry, targets)
#                 loss = sharp_weight * loss_sharp + blurry_weight * loss_blurry

#                 val_losses.append(loss.item())
#                 val_sharp_losses.append(loss_sharp.item())
#                 val_blurry_losses.append(loss_blurry.item())

#         avg_val_loss        = np.mean(val_losses)
#         avg_val_sharp_loss  = np.mean(val_sharp_losses)
#         avg_val_blurry_loss = np.mean(val_blurry_losses)

#         log_dict['validation_loss_per_epoch'].append(avg_val_loss)
#         log_dict['validation_sharp_loss_per_epoch'].append(avg_val_sharp_loss)
#         log_dict['validation_blurry_loss_per_epoch'].append(avg_val_blurry_loss)

#         print(
#             f"Epoch {epoch+1}: "
#             f"Train {avg_train_loss:.4f} (sharp {avg_train_sharp_loss:.4f} | blurry {avg_train_blurry_loss:.4f}) | "
#             f"Val {avg_val_loss:.4f} (sharp {avg_val_sharp_loss:.4f} | blurry {avg_val_blurry_loss:.4f})"
#         )

#         if scheduler:
#             scheduler.step(avg_val_loss)

#         if (epoch + 1) % save_every_epochs == 0:
#             save_model_and_logs(
#                 model=model,
#                 optimizer=optimizer,
#                 save_path=save_path,
#                 weights_name=weights_name,
#                 log_dict=log_dict,
#                 epoch=epoch,
#                 scheduler=scheduler,
#                 current_max_noise=max_noise,
#                 config={
#                     "initial_lr": optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"]),
#                     "current_lr": optimizer.param_groups[0]["lr"],
#                     "image_size": image_size,
#                     "n_channels": n_channels,
#                     "sharp_weight": sharp_weight,
#                     "blurry_weight": blurry_weight,
#                 },
#                 noise_config={
#                     "initial_max_noise": initial_max_noise,
#                     "n_reduce_factor": n_reduce_factor,
#                     "reduce_on": reduce_on,
#                 },
#             )

#     save_model_and_logs(
#         model=model,
#         optimizer=optimizer,
#         save_path=save_path,
#         weights_name=weights_name,
#         log_dict=log_dict,
#         epoch=epoch,
#         scheduler=scheduler,
#         current_max_noise=max_noise,
#         config={
#             "initial_lr": optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"]),
#             "current_lr": optimizer.param_groups[0]["lr"],
#             "image_size": image_size,
#             "n_channels": n_channels,
#             "sharp_weight": sharp_weight,
#             "blurry_weight": blurry_weight,
#         },
#         noise_config={
#             "initial_max_noise": initial_max_noise,
#             "n_reduce_factor": n_reduce_factor,
#             "reduce_on": reduce_on,
#         },
#     )

#     return log_dict

def train_SR_head_autoencoder(model, optimizer, loss_function, train_loader, val_loader,
                      device, epochs=3000, save_every_epochs=100, image_size=(64, 64), n_channels=3,
                      scheduler=None, noise_fn=None, initial_max_noise=0.16,
                      n_reduce_factor=0.5, reduce_on=1000, save_path=None, weights_name=None,
                      # New: weights for the two loss terms
                      sharp_weight=1.0, blurry_weight=0.5):
    """ Train a super-res autoencoder with or without skip connections. Model returns (sharp, blurry). """

    log_dict = {
        'training_loss_per_epoch': [],
        'validation_loss_per_epoch': [],
        # Extra: track the two components so you can see which is driving training
        'training_sharp_loss_per_epoch': [],
        'training_blurry_loss_per_epoch': [],
        'validation_sharp_loss_per_epoch': [],
        'validation_blurry_loss_per_epoch': [],
    }

    model.to(device)
    max_noise = initial_max_noise

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        if (epoch + 1) % reduce_on == 0:
            max_noise *= n_reduce_factor

        # --- Training ---
        model.train()
        train_losses, train_sharp_losses, train_blurry_losses = [], [], []

        for images in tqdm(train_loader, desc="Training"):
            optimizer.zero_grad()
            images = images[0][:, 0:n_channels, ...].to(device)
            noisy_images = noise_fn(images, max_val=max_noise) if noise_fn else images

            targets = images.view(-1, n_channels, *image_size)

            # Forward pass — now returns two outputs
            sharp, blurry = model(noisy_images)

            loss_sharp  = loss_function(sharp,  targets)
            loss_blurry = loss_function(blurry, targets)
            loss = sharp_weight * loss_sharp + blurry_weight * loss_blurry

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_sharp_losses.append(loss_sharp.item())
            train_blurry_losses.append(loss_blurry.item())

        avg_train_loss        = np.mean(train_losses)
        avg_train_sharp_loss  = np.mean(train_sharp_losses)
        avg_train_blurry_loss = np.mean(train_blurry_losses)

        log_dict['training_loss_per_epoch'].append(avg_train_loss)
        log_dict['training_sharp_loss_per_epoch'].append(avg_train_sharp_loss)
        log_dict['training_blurry_loss_per_epoch'].append(avg_train_blurry_loss)

        # --- Validation ---
        model.eval()
        val_losses, val_sharp_losses, val_blurry_losses = [], [], []

        with torch.no_grad():
            for val_images in tqdm(val_loader, desc="Validating"):
                val_images = val_images[0][:, 0:n_channels, ...].to(device)
                targets = val_images.view(-1, n_channels, *image_size)

                sharp, blurry = model(val_images)

                loss_sharp  = loss_function(sharp,  targets)
                loss_blurry = loss_function(blurry, targets)
                loss = sharp_weight * loss_sharp + blurry_weight * loss_blurry

                val_losses.append(loss.item())
                val_sharp_losses.append(loss_sharp.item())
                val_blurry_losses.append(loss_blurry.item())

        avg_val_loss        = np.mean(val_losses)
        avg_val_sharp_loss  = np.mean(val_sharp_losses)
        avg_val_blurry_loss = np.mean(val_blurry_losses)

        log_dict['validation_loss_per_epoch'].append(avg_val_loss)
        log_dict['validation_sharp_loss_per_epoch'].append(avg_val_sharp_loss)
        log_dict['validation_blurry_loss_per_epoch'].append(avg_val_blurry_loss)

        print(
            f"Epoch {epoch+1}: "
            f"Train {avg_train_loss:.4f} (sharp {avg_train_sharp_loss:.4f} | blurry {avg_train_blurry_loss:.4f}) | "
            f"Val {avg_val_loss:.4f} (sharp {avg_val_sharp_loss:.4f} | blurry {avg_val_blurry_loss:.4f})"
        )

        if scheduler:
            scheduler.step(avg_val_loss)

        if (epoch + 1) % save_every_epochs == 0:
            save_model_and_logs(
                model=model,
                optimizer=optimizer,
                save_path=save_path,
                weights_name=weights_name,
                log_dict=log_dict,
                epoch=epoch,
                scheduler=scheduler,
                current_max_noise=max_noise,
                config={
                    "initial_lr": optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"]),
                    "current_lr": optimizer.param_groups[0]["lr"],
                    "image_size": image_size,
                    "n_channels": n_channels,
                    "sharp_weight": sharp_weight,
                    "blurry_weight": blurry_weight,
                },
                noise_config={
                    "initial_max_noise": initial_max_noise,
                    "n_reduce_factor": n_reduce_factor,
                    "reduce_on": reduce_on,
                },
            )

    save_model_and_logs(
        model=model,
        optimizer=optimizer,
        save_path=save_path,
        weights_name=weights_name,
        log_dict=log_dict,
        epoch=epoch,
        scheduler=scheduler,
        current_max_noise=max_noise,
        config={
            "initial_lr": optimizer.param_groups[0].get("initial_lr", optimizer.param_groups[0]["lr"]),
            "current_lr": optimizer.param_groups[0]["lr"],
            "image_size": image_size,
            "n_channels": n_channels,
            "sharp_weight": sharp_weight,
            "blurry_weight": blurry_weight,
        },
        noise_config={
            "initial_max_noise": initial_max_noise,
            "n_reduce_factor": n_reduce_factor,
            "reduce_on": reduce_on,
        },
    )

    return log_dict