extends Node

@export var gakken: Node3D
@export var camera: Node3D

var relative_pos: Vector3

func _ready():
	relative_pos = camera.position - gakken.position;
	# TODO: Init necessary camera tracking server/hardware

func _process(delta):
	relative_pos = get_relative_camera_position()
	camera.position = gakken.position + relative_pos

func get_relative_camera_position() -> Vector3:
	# TODO: get relative position from tracking camera
	return relative_pos # TODO: and remove this line
