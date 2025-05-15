def _check_config(config):

    # Telescope
    if "diameter" not in config["telescope"]:
        raise ValueError("diameter is mandatory")
    
    if "central_obscuration" not in config["telescope"]:
        config["telescope"]["central_obscuration"] = 0.0

    if "spider" not in config["telescope"]:
        config["telescope"]["spider"] = 0

    # Images
    if "n_pixel" not in config["images"]:
        raise ValueError("n_pixel is mandatory")
    
    if "pix_size" not in config["images"]:
        raise ValueError("pix_size is mandatory")
    
    if "apodization_border" not in config["images"]:
        config["images"]["apodization_border"] = 0
    
    if "remove_gradient_apodization" not in config["images"]:
        config["images"]["remove_gradient_apodization"] = False

    # Optimization
    if "gpu" not in config["optimization"]:
        config["optimization"]["gpu"] = -1

    if "transform" not in config["optimization"]:
        config["optimization"]["gpu"] = "none"

    if "softplus_scale" not in config["optimization"]:
        config["optimization"]["softplus_scale"] = 1.0
    
    if "lr_obj" not in config["optimization"]:
        raise ValueError("lr_obj mandatory")
    
    if "lr_modes" not in config["optimization"]:
        raise ValueError("lr_modes is mandatory")
        
    if config["optimization"]["transform"] not in ["softplus", "none"]:
        raise ValueError(f"Invalid value for transform. It is {config['optimization']['transform']} but should be softplus or none")
    
    if config["initialization"]["object"] not in ["contrast", "average"]:
        raise ValueError(f"Invalid value for initialization of the object. It is {config['initialization']['object']} but should be contrast or average")
    
    if config["annealing"]["type"] not in ["sigmoid", "linear", "none"]:
        raise ValueError(f"Invalid value for annealing type. It is {config['annealing']['type']} but should be sigmoid, linear or none")
    
    if config["psf"]["model"] not in ["zernike", "kl", "pca", "nmf"]:
        raise ValueError(f"Invalid value for psf model. It is {config['psf']['model']} but should be zernike, kl, pca or nmf")
    
    for k, v in config.items():
        if "object" in k:
            if v["image_filter"] not in ["tophat", "scharmer", "none"]:
                raise ValueError(f"Invalid value for image_filter. It is {v['image_filter']} but should be tophat, scharmer or none")
                                    
    return config