@tool
extends ColorRect

@export var gakken: Node3D
@export var camera: Camera3D


func _process(_delta):
	var mvp_matrix = camera.get_camera_projection()
	mvp_matrix *= Projection(camera.get_camera_transform()).inverse()
	mvp_matrix *= Projection(gakken.transform)

	# set shader mvp_matrix
	var shadermat = material as ShaderMaterial
	shadermat.set_shader_parameter("mvp_matrix", mvp_matrix)
