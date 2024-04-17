import vizdoom as vzd


def game_init(config_file_path, mode='train'):
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.set_mode(vzd.Mode.PLAYER)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Adjust if needed
    game.set_available_game_variables([
        vzd.GameVariable.AMMO2,
        vzd.GameVariable.HEALTH,
        vzd.GameVariable.KILLCOUNT,
        vzd.GameVariable.POSITION_X,
        vzd.GameVariable.POSITION_Y,
        vzd.GameVariable.POSITION_Z,
        vzd.GameVariable.FRAGCOUNT,
        vzd.GameVariable.ARMOR,
        vzd.GameVariable.DEAD,
        vzd.GameVariable.DEATHCOUNT,
        vzd.GameVariable.ITEMCOUNT,
        vzd.GameVariable.SECRETCOUNT,
        vzd.GameVariable.HITCOUNT,
        vzd.GameVariable.DAMAGECOUNT,
    ])

    game.set_available_buttons([
        vzd.Button.MOVE_LEFT,
        vzd.Button.MOVE_RIGHT,
        vzd.Button.ATTACK,
        vzd.MOVE_FORWARD,
        vzd.MOVE_BACKWARD,
        vzd.TURN_LEFT,
        vzd.TURN_RIGHT,
    ])

    if mode == 'train':
        game.set_window_visible(False)
    if mode == 'test':
        game.set_window_visible(True)
    game.init()
    return game
