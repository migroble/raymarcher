#define MAX_ITER 1000
#define MAX_DIST 10.0f
#define EPSILON 0.0001f
#define FOV 90
#define POWER 8.0f
#define MANDEL_ITER 100
#define v3 (float3)

float2 rot2d(float2 p, float theta) {
    float cos_theta;
    float sin_theta = sincos(theta, &cos_theta);
    return (float2)(p.x * cos_theta - p.y * sin_theta, p.x * sin_theta + p.y * cos_theta);
}

float3 rot_x(float3 p, float theta) {
    return v3(p.x, rot2d(p.yz, theta));
}

float3 rot_y(float3 p, float theta) {
    float2 q = rot2d(p.xz, theta);
    return v3(q.x, p.y, q.y);
}

float3 rot_z(float3 p, float theta) {
    return v3(rot2d(p.xy, theta), p.z);
}

float3 rot(float3 p, float3 theta) {
    return rot_z(rot_y(rot_x(p, theta.x), theta.y), theta.z);
}

float op_union(float d1, float d2) {
    return fmin(d1, d2);
}

float op_substraction(float d1, float d2) {
    return fmax(-d1, d2);
}

float op_intersect(float d1, float d2) {
    return fmax(d1, d2);
}

float op_smooth_union(float d1, float d2, float k) {
    float h = clamp(0.5f + 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return mix(d2, d1, h) - k * h * (1.0f - h);
}

float op_smooth_substraction(float d1, float d2, float k) {
    float h = clamp(0.5f - 0.5f * (d2 + d1) / k, 0.0f, 1.0f);
    return mix(d2, -d1, h) - k * h * (1.0f - h);
}

float op_smooth_intersect(float d1, float d2, float k) {
    float h = clamp(0.5f - 0.5f * (d2 - d1) / k, 0.0f, 1.0f);
    return mix(d2, d1, h) - k * h * (1.0f - h);
}

float3 reflect(float3 d, float3 n) {
    return 2 * dot(d, n) * n - d;
}

float sdf_box(float3 p, float3 b) {
    float3 q = fabs(p) - b;
    return length(fmax(q, (float3)0.0f)) + fmin(fmax(q.x, fmax(q.y, q.z)), 0.0f);
}

float sdf_sphere(float3 p, float r) {
    return length(p) - r;
}

float sdf_mandelbulb(float3 p) {
    float3 z = p;
    float dr = 1.0;
    float r = 0.0;
    
    for (uint i = 0; i < MANDEL_ITER; ++i) {
        r = length(z);
        
        if (r > 2.0f)
            break;
        
        float theta = acos(z.z / r);
        float phi = atan2(z.y, z.x);
        dr =  pow(r, POWER - 1.0f) * POWER * dr + 1.0f;
        
        float zr = pow(r, POWER);
        theta = theta * POWER;
        phi = phi * POWER;
        
        float cos_theta;
        float sin_theta = sincos(theta, &cos_theta);
        float cos_phi;
        float sin_phi = sincos(phi, &cos_phi);
        
        z = zr * (float3)(sin_theta * cos_phi, sin_phi * sin_theta, cos_theta);
        z += p;
    }
    
    return 0.5f * log(r) * r / dr;
}

float sdf_scene(float3 p) {
    return sdf_mandelbulb(rot_x(p - v3(0.0f, 0.0f, 0.6f), M_PI_2));
}

float3 get_normal(float3 p) {
    const float2 k = (float2)(1.0f, -1.0f);
    
    return normalize(
        k.xyy * sdf_scene(p + k.xyy * EPSILON) + 
        k.yyx * sdf_scene(p + k.yyx * EPSILON) + 
        k.yxy * sdf_scene(p + k.yxy * EPSILON) + 
        k.xxx * sdf_scene(p + k.xxx * EPSILON)
    );
}

float raymarch(float3 eye, float3 ray_dir, uint *steps) {
    float dist;
    float depth = 0.0f;
    *steps = 0;
    
    for (uint i = 0; i < MAX_ITER; ++i) {
        dist = sdf_scene(eye + depth * ray_dir);
        
        ++*steps;
        
        if (dist < EPSILON)
            return depth;
        
        depth += dist;
        
        if (depth > MAX_DIST)
            return MAX_DIST;
    }
    
    return MAX_DIST;
}

float calculate_shadow(float3 hit, float3 ray_dir) {
    uint steps = 0;
    float shadow_intensity = 0.2f;
    float brightness = 1.0f;
    float dist;
    float depth = 0.01f;
    
    for (uint i = 0; i < MAX_ITER; ++i) {
        dist = sdf_scene(hit + depth * ray_dir);
        
        if (dist < EPSILON)
            return shadow_intensity;
        
        brightness = min(brightness, dist * 200.0f);
        
        depth += dist;
    }
    
    return shadow_intensity + (1.0f - shadow_intensity) * brightness;
}

float3 glow(float intensity, float3 glow_color, uint steps) {
    return log(1.0f + intensity * (float)steps / (float)MAX_ITER) * glow_color;
}

kernel void render(global unsigned char *image, const uint width, const uint height) {
    uint idx = get_global_id(0);
    uint idy = get_global_id(1);
    
    float w = get_global_size(0);
    float h = get_global_size(1);
    
    float m = 2.0f * tanpi((float)FOV / 360.0f) / h;
    float x = m * (idx - w / 2);
    float y = m * (idy - h / 2);
    
    float3 color = v3(0.0f);
    float3 eye   = v3(0.0f, 0.0f, -1.0f);
    float3 ray   = normalize(v3(x, y, 0.0f) - eye);
    
    float3 light_dir   = normalize(v3(0.75f, -1.0f, -0.5f));
    float3 light_color = v3(0.55f, 0.1f, 0.7f);
    
    float3 glow_color  = v3(0.7f, 0.15f, 0.6f);
    
    uint steps;
    float depth;
    if ((depth = raymarch(eye, ray, &steps)) != MAX_DIST) {
        float3 p = eye + depth * ray;
        float3 normal = get_normal(p);
        
        float ambient  = 0.05f;
        float diffuse  = fmax(dot(normal, light_dir), 0.0f);
        float shadow   = calculate_shadow(p, -light_dir);
        float specular = pow(fmax(dot(reflect(light_dir, normal), -ray), 0.0f), 256.0f);
        
        color = clamp(diffuse * shadow + specular, ambient, 1.0f) * light_color;
    }
    
    color += glow(5.0f, glow_color, steps);
    color = fmin(color, v3(1.0f));
    
    image[3 * (idx + idy * width) + 0] = (unsigned char)(255.0 * color.x);
    image[3 * (idx + idy * width) + 1] = (unsigned char)(255.0 * color.y);
    image[3 * (idx + idy * width) + 2] = (unsigned char)(255.0 * color.z);
}
