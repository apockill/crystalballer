extends MeshInstance3D

@export var look_interval_min = 0.1
@export var look_interval_max = 2
@export var look_speed = 0.1

var look_timer = 0;
var look_progress = 1;
var last_look = Vector3(0, 0, 0);
var current_look = Vector3(0, 0, 0);

func _ready():
	look_timer = randf_range(look_interval_min, look_interval_max)

func _process(delta):
	if look_timer > 0:
		look_timer -= delta
		if look_timer <= 0:
			look_progress = 0
			last_look = current_look
			var x_look = deg_to_rad(randf_range(-50, 50))
			var z_look = deg_to_rad(randf_range(-50, 50))
			current_look = Vector3(x_look, 0, z_look)
	else:
		look_progress += delta / look_speed
		if look_progress >= 1:
			look_progress = 1
			look_timer = randf_range(look_interval_min, look_interval_max)
		rotation.x = lerpf(last_look.x, current_look.x, look_progress)
		rotation.z = lerpf(last_look.z, current_look.z, look_progress)
