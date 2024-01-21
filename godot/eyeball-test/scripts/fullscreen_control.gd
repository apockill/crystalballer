extends Node

var fullscreen = false


func _process(delta):
	if Input.is_action_just_pressed("fullscreen_switch"):
		if fullscreen:
			fullscreen = false
			DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_WINDOWED)
		else:
			fullscreen = true
			DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_FULLSCREEN)
