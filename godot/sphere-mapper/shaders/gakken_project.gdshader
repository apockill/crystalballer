shader_type canvas_item;

uniform mat4 mvp_matrix;
uniform float gakken_radius = 0.5;
uniform float distance_ratio = 1.6666;
uniform sampler2D screen_texture : hint_screen_texture;

vec2 project(vec2 v1, vec2 v2) {
	return dot(v1, v2) * v1;
}

void fragment() {
	// calculate sphere xy position
	// convert coordinates so 0,0 is at the center of screen
	float aspect_ratio = SCREEN_PIXEL_SIZE.x / SCREEN_PIXEL_SIZE.y;
	vec2 project_pos = vec2(
		(SCREEN_UV.x - 0.5) * 2.0 / aspect_ratio,
		(SCREEN_UV.y - 0.5) * 2.0
	);
	
	float project_dist = length(project_pos.xy);
	if (project_dist > 1.0) {
		COLOR = vec4(0, 0, 0, 1);
	} else {
		// calculate sphere raycast collision
		// https://www.desmos.com/calculator/rlvaxyy7js
		
		// casting can be done more simply in 2d
		// create a 2d casting dir that will find the circle intersection
		float sphere_dist = gakken_radius * distance_ratio;
		vec2 cast_vec = normalize(vec2(
			sphere_dist,
			gakken_radius * project_dist
		));
		
		// create vector to center of the sphere
		vec2 sphere_origin = vec2(sphere_dist, 0);
		
		// project sphere center vector onto casting vector
		vec2 cast_align = project(cast_vec, sphere_origin);
		
		// get the distance from the aligned cast to the center of the circle
		float align_dist = length(sphere_origin - cast_align);
		
		// get the dist from the aligned cast to the outer edge of the circle
		float align_intersect_dist = sqrt(
			(gakken_radius * gakken_radius) -
			(align_dist * align_dist)
		);
		
		// calculate the dist until hit for the outer edge of the circle
		float hit_dist = length(cast_align + (align_intersect_dist * cast_vec));
		
		// create 3d vector to represent ray cast
		vec3 sphere_pos = normalize(vec3(
			gakken_radius * project_pos.x,
			sphere_dist,
			gakken_radius * project_pos.y
		)) * hit_dist;
		
		// reduce the y axis by the sphere distance for normalization
		sphere_pos.y -= sphere_dist;

		// extract uv coordinate using mvp matrix
		vec4 clip_pos = mvp_matrix * vec4(sphere_pos, 1);
		vec3 ndc_pos = clip_pos.xyz / clip_pos.w;
		vec2 uv_pos = vec2(
			(ndc_pos.x + 1.0) / 2.0,
			(-ndc_pos.y + 1.0) / 2.0
		);
		
		COLOR = vec4(sphere_pos.xyz, 1);
		COLOR = textureLod(screen_texture, uv_pos, 0);
	}
}
