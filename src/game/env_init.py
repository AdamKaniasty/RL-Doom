import vizdoom as vzd


def game_init(config_file_path, mode='train'):
    """
    Initialize the Vizdoom game with the given configuration file path (scenario) and mode.
    Used in gymnasium_wrapper.py to create the environment.
    """

    print("Initializing game with config using: 'src/game/env_init.py' ")

    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_death_penalty(5)
    game.set_render_crosshair(False)
    game.set_render_crosshair(True)

    game.set_doom_skill(5)  # To powinno być zmienione w samym pliku cfg a nie tu

    # Adjust if needed
    # Only health, position , killcount, damage_taken are relevant for corridor scenario
    game.set_available_game_variables([
        vzd.GameVariable.AMMO2,
        vzd.GameVariable.HEALTH,
        vzd.GameVariable.KILLCOUNT,
        vzd.GameVariable.POSITION_X,
        # vzd.GameVariable.POSITION_Y,
        # vzd.GameVariable.POSITION_Z,
        # vzd.GameVariable.FRAGCOUNT,
        # vzd.GameVariable.ARMOR,
        # vzd.GameVariable.DEAD,
        # vzd.GameVariable.DEATHCOUNT,
        # vzd.GameVariable.ITEMCOUNT,
        # vzd.GameVariable.SECRETCOUNT,
        # vzd.GameVariable.HITCOUNT,
        # vzd.GameVariable.DAMAGECOUNT,
        # vzd.GameVariable.DAMAGE_TAKEN,
    ])

    game.set_available_buttons([
        vzd.TURN_LEFT,
        vzd.TURN_RIGHT,
        vzd.Button.ATTACK,
    ])

    if mode == 'train':
        game.set_window_visible(False)
    if mode == 'test':
        game.set_window_visible(True)
    return game
