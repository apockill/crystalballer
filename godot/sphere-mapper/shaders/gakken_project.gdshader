shader_type canvas_item;

uniform vec3 gakken_dir = vec3(0, 1, 0);
uniform vec3 camera_dir = vec3(0, -1, 0);
uniform sampler2D screen_texture : hint_screen_texture;

void fragment() {
	float aspect_ratio = SCREEN_PIXEL_SIZE.x / SCREEN_PIXEL_SIZE.y;
	vec2 screen_pos = vec2(
		(SCREEN_UV.x - 0.5) * 2.0 / aspect_ratio,
		(SCREEN_UV.y - 0.5) * 2.0
	);
	
	COLOR = vec4(0, 0, 0, 1);
	if (length(screen_pos) <= 1.0) {
		COLOR = textureLod(screen_texture, SCREEN_UV, 0);
	}
}