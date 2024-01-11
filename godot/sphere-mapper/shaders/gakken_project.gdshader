shader_type canvas_item;

uniform vec3 camera_up = vec3(0, 1, 0);
uniform vec3 camera_right = vec3(1, 0, 0);
uniform sampler2D screen_texture : hint_screen_texture;

void fragment() {
	// calculate sphere xy position
	// convert coordinates so 0,0 is at the center of screen
	float aspect_ratio = SCREEN_PIXEL_SIZE.x / SCREEN_PIXEL_SIZE.y;
	vec3 sphere_pos = vec3(
		(SCREEN_UV.x - 0.5) / aspect_ratio,
		(SCREEN_UV.y - 0.5),
		0
	);
	
	float dist = length(sphere_pos.xy);
	if (dist > 0.5) {
		COLOR = vec4(0, 0, 0, 1);
	} else {
		// calculate sphere z position
		// https://www.desmos.com/calculator/apdgozwv7g
		sphere_pos.z = max(0, sqrt(0.25 - (dist * dist)));

		// extract uv coordinate using camera up and right vectors
		// convert coordinates back to UV space with 0,0 in upper left
		vec2 texUV = vec2(
			(dot(sphere_pos, camera_right) * aspect_ratio + 0.5),
			(dot(sphere_pos, camera_up) + 0.5)
		);
		
		COLOR = textureLod(screen_texture, texUV, 0);
	}
}