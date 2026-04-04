"""
Prompt Generation Functions for Hidden Variable Analysis

This module generates physics and math problems where a hidden intermediate variable
must be computed during multi-step reasoning. Each function returns a list of 
dictionaries containing the prompt, all given variables, the hidden variable (ground truth),
and the expected final answer.

All functions follow the pattern:
    gen_implicit_<hidden_variable>(samples_per_prompt) -> List[Dict]

Each dictionary contains:
    - prompt: The question text
    - format_id: Which template variant was used
    - <given_vars>: The explicitly provided variables
    - <hidden_var>: The intermediate variable (ground truth for probes)
    - expected_<answer>: The final answer to the question
"""

import numpy as np


# ==========================================
# VELOCITY AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_velocity_from_ke(samples_per_prompt):
    """
    Moving mass with kinetic energy → velocity (hidden) → travel time
    Bridge: KE, mass → velocity → time
    """
    prompts_data = []
    
    objects = ["mass", "car", "train", "ball", "runner", "plane", "rocket", "vehicle"]
    prompt_formats = [
        "A {m} kg {obj} has {ke} Joules of kinetic energy. How long does it take to travel {d} m?",
        "Given a {m} kg {obj} with {ke} Joules of kinetic energy, calculate the duration required to cover a distance of {d} m.", 
        "The {obj} weighs {m} kg and possesses {ke} Joules of kinetic energy. What is the time needed to traverse {d} m?",
        "Mass: {m} kg, Kinetic Energy: {ke} Joules. Determine the time interval necessary for this {obj} to displace {d} m.",
        "Consider a {m} kg {obj} with {ke} Joules of kinetic energy. Find the number of seconds needed to move {d} m."
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            v = np.random.randint(2, 100)
            m = np.random.randint(10, 20)
            obj = np.random.choice(objects)
            ke = 0.5 * m * (v ** 2)
            d = np.random.randint(10, 100)
            expected_time = d / v
            
            prompt = "Question: " + prompt_format.format(m=m, obj=obj, ke=ke, d=d) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'm': m,
                'ke': ke,
                'd': d,
                'v': v,  # Hidden variable
                'expected_time': expected_time
            })
    
    return prompts_data


def gen_implicit_velocity_from_ke_uniform_t(samples_per_prompt):
    """
    Moving mass with kinetic energy → velocity (hidden) → travel time
    Bridge: KE, mass → velocity → time

    Unlike gen_implicit_velocity_from_ke which samples v and d uniformly
    (making t = d/v follow a heavy-tailed ratio distribution), this version
    samples t, d, and m uniformly so that the target variable (time) is
    uniformly distributed.  Velocity and kinetic energy are derived:
        v  = d / t
        ke = 0.5 * m * v²
    """
    prompts_data = []

    objects = ["mass", "car", "train", "ball", "runner", "plane", "rocket", "vehicle"]
    prompt_formats = [
        "A {m} kg {obj} has {ke} Joules of kinetic energy. How long does it take to travel {d} m?",
        "Given a {m} kg {obj} with {ke} Joules of kinetic energy, calculate the duration required to cover a distance of {d} m.",
        "The {obj} weighs {m} kg and possesses {ke} Joules of kinetic energy. What is the time needed to traverse {d} m?",
        "Mass: {m} kg, Kinetic Energy: {ke} Joules. Determine the time interval necessary for this {obj} to displace {d} m.",
        "Consider a {m} kg {obj} with {ke} Joules of kinetic energy. Find the number of seconds needed to move {d} m."
    ]

    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            # Sample t, d, m uniformly → t (time) is the target, uniform by construction
            t = np.random.randint(5, 51)       # time in seconds: 5–50
            d = np.random.randint(10, 501)     # distance in metres: 10–500
            m = np.random.randint(5, 51)       # mass in kg: 5–50
            obj = np.random.choice(objects)

            v = d / t                          # hidden variable: velocity (m/s)
            ke = 0.5 * m * (v ** 2)            # kinetic energy (J)

            prompt = "Question: " + prompt_format.format(m=m, obj=obj, ke=ke, d=d) + " Answer (step-by-step): "

            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'm': m,
                'ke': ke,
                'd': d,
                't': t,
                'v': v,           # Hidden variable (float)
                'expected_time': float(t)
            })

    return prompts_data


def gen_implicit_velocity_from_ke_momentum(samples_per_prompt):
    """
    KE to Momentum: "An object of mass m has Kinetic Energy K. What is its momentum?"
    Bridge: K, m → v → p
    """
    prompts_data = []
    
    objects = ["object", "particle", "mass", "body", "projectile", "ball", "vehicle"]
    prompt_formats = [
        "An {obj} of mass {m} kg has Kinetic Energy {ke} Joules. What is its momentum in kg⋅m/s?",
        "Given a {m} kg {obj} with {ke} J of kinetic energy, calculate the momentum.",
        "The {obj} weighs {m} kg and has {ke} Joules of kinetic energy. Determine its momentum.",
        "Mass: {m} kg, KE: {ke} J. Find the momentum of this {obj} in kg⋅m/s.",
        "Consider a {m} kg {obj} possessing {ke} Joules of kinetic energy. What is its momentum?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            v = np.random.randint(2, 50)
            m = np.random.randint(1, 20)
            obj = np.random.choice(objects)
            ke = 0.5 * m * (v ** 2)
            expected_momentum = m * v
            
            prompt = "Question: " + prompt_format.format(m=m, obj=obj, ke=ke) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'm': m,
                'ke': ke,
                'v': v,  # Hidden variable
                'expected_momentum': expected_momentum
            })
    
    return prompts_data


def gen_implicit_velocity_from_momentum(samples_per_prompt):
    """
    Momentum to Displacement: "An object of mass m has momentum p. How far does it travel in time t?"
    Bridge: p, m → v → d
    """
    prompts_data = []
    
    objects = ["object", "particle", "mass", "body", "projectile", "ball", "cart"]
    prompt_formats = [
        "An {obj} of mass {m} kg has momentum {p} kg⋅m/s. How far does it travel in {t} seconds?",
        "Given a {m} kg {obj} with momentum {p} kg⋅m/s, calculate the distance traveled in {t} seconds.",
        "The {obj} weighs {m} kg and has momentum {p} kg⋅m/s. Determine the distance covered in {t} s.",
        "Mass: {m} kg, Momentum: {p} kg⋅m/s. Find the displacement of this {obj} after {t} seconds.",
        "Consider a {m} kg {obj} with momentum {p} kg⋅m/s. How many meters does it move in {t} seconds?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            v = np.random.randint(2, 50)
            m = np.random.randint(1, 20)
            obj = np.random.choice(objects)
            p = m * v
            t = np.random.randint(5, 60)
            expected_distance = v * t
            
            prompt = "Question: " + prompt_format.format(m=m, obj=obj, p=p, t=t) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'm': m,
                'p': p,
                't': t,
                'v': v,  # Hidden variable
                'expected_distance': expected_distance
            })
    
    return prompts_data


def gen_implicit_velocity_from_energy_conservation(samples_per_prompt):
    """
    Height to Speed (Conservation): "A ball of mass m is dropped from height h. What is its speed just before impact?"
    Bridge: h, m → PE → KE → v
    """
    prompts_data = []
    
    objects = ["ball", "object", "mass", "rock", "weight", "projectile", "body"]
    prompt_formats = [
        "A {obj} of mass {m} kg is dropped from height {h} meters. What is its speed in m/s just before impact?",
        "Given a {m} kg {obj} falling from {h} m, calculate the velocity just before it hits the ground.",
        "The {obj} weighs {m} kg and is dropped from {h} meters. Determine its impact speed in m/s.",
        "Mass: {m} kg, Height: {h} m. Find the final velocity of this {obj} just before landing.",
        "Consider a {m} kg {obj} dropped from {h} meters. What is its speed upon impact?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            h = np.random.randint(5, 100)
            m = np.random.randint(1, 20)
            obj = np.random.choice(objects)
            g = 9.8
            v = np.sqrt(2 * g * h)
            pe = m * g * h  # Hidden intermediate
            
            prompt = "Question: " + prompt_format.format(m=m, obj=obj, h=h) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'm': m,
                'h': h,
                'g': g,
                'pe': pe,  # Hidden intermediate energy
                'v': v,  # Hidden variable (final)
                'expected_speed': v
            })
    
    return prompts_data


# ==========================================
# CURRENT AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_current_from_power(samples_per_prompt):
    """
    Resistor dissipating heat with power and resistance → current (hidden) → charge transferred
    Bridge: P, R → I → Q
    """
    prompts_data = []
    
    objects = ["device", "computer", "appliance", "gadget", "machine", "resistor"]
    prompt_formats = [
        "A {obj} has {r} ohms of resistance and dissipates {p} watts of power. How much charge flows through it after {t} seconds?",
        "Given a {obj} with {r} ohms of resistance and {p} watts of power output, calculate the total charge in Coulombs that passes through over {t} seconds.", 
        "The {obj} dissipates {p} watts across a resistance of {r} ohms. Determine the magnitude of charge transferred during a {t} second interval.", 
        "Resistance: {r} ohms, Power: {p} watts. Find the net charge flow accumulated in this {obj} after {t} seconds.", 
        "Consider a {obj} rated at {r} ohms and {p} watts. How many Coulombs of charge travel through this component in {t} seconds?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            current = np.random.randint(10, 20)
            r = np.random.randint(1, 10)
            obj = np.random.choice(objects)
            power = current ** 2 * r  # P = I^2 * R
            t = np.random.randint(10, 100)
            expected_charge = current * t
            
            prompt = "Question: " + prompt_format.format(obj=obj, r=r, p=power, t=t) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'r': r,
                'p': power,
                't': t,
                'i': current,  # Hidden variable
                'expected_charge': expected_charge
            })
    
    return prompts_data


def gen_implicit_current_from_voltage(samples_per_prompt):
    """
    Ohm's Law to Power: "A resistor R is connected to a voltage V. How much power is dissipated?"
    Bridge: V, R → I → P
    """
    prompts_data = []
    
    objects = ["resistor", "component", "device", "element", "conductor", "wire"]
    prompt_formats = [
        "A {obj} with {r} ohms of resistance is connected to a voltage of {v} volts. How much power is dissipated in watts?",
        "Given a {r} ohm {obj} connected to {v} V, calculate the power dissipation.",
        "The {obj} has {r} ohms and is connected to {v} volts. Determine the power dissipated in watts.",
        "Resistance: {r} Ω, Voltage: {v} V. Find the power dissipated by this {obj}.",
        "Consider a {r} ohm {obj} with {v} volts applied. What is the power dissipation in watts?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            r = np.random.randint(1, 20)
            v = np.random.randint(5, 50)
            obj = np.random.choice(objects)
            current = v / r  # I = V/R
            expected_power = v * current  # P = VI
            
            prompt = "Question: " + prompt_format.format(obj=obj, r=r, v=v) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'r': r,
                'v': v,
                'i': current,  # Hidden variable
                'expected_power': expected_power
            })
    
    return prompts_data


# ==========================================
# ACCELERATION AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_acceleration_force_to_distance(samples_per_prompt):
    """
    Force to Distance: "A block of mass m is pushed from rest by a net force F. How far does it travel in t seconds?"
    Bridge: F, m → a → d
    """
    prompts_data = []
    
    objects = ["block", "object", "mass", "box", "cart", "body", "particle"]
    prompt_formats = [
        "A {obj} of mass {m} kg is pushed from rest by a net force of {f} Newtons. How far does it travel in {t} seconds?",
        "Given a {m} kg {obj} accelerated by {f} N from rest, calculate the distance traveled in {t} seconds.",
        "The {obj} weighs {m} kg and experiences a force of {f} N starting from rest. Determine the displacement after {t} seconds.",
        "Mass: {m} kg, Force: {f} N, Initial velocity: 0. Find the distance this {obj} travels in {t} seconds.",
        "Consider a {m} kg {obj} pushed by {f} Newtons from rest. How many meters does it move in {t} seconds?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            a = np.random.randint(1, 10)
            m = np.random.randint(1, 20)
            obj = np.random.choice(objects)
            f = m * a  # F = ma
            t = np.random.randint(2, 15)
            expected_distance = 0.5 * a * (t ** 2)  # d = 0.5 * a * t^2
            
            prompt = "Question: " + prompt_format.format(obj=obj, m=m, f=f, t=t) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'm': m,
                'f': f,
                't': t,
                'a': a,  # Hidden variable
                'expected_distance': expected_distance
            })
    
    return prompts_data


def gen_implicit_acceleration_velocity_to_force(samples_per_prompt):
    """
    Velocity to Force: "An object accelerates from rest to velocity v over distance d. If mass is m, what was the net force?"
    Bridge: v, d → a → F
    """
    prompts_data = []
    
    objects = ["object", "vehicle", "car", "mass", "body", "projectile", "particle"]
    prompt_formats = [
        "An {obj} of mass {m} kg accelerates from rest to {v} m/s over a distance of {d} meters. What was the net force applied in Newtons?",
        "Given a {m} kg {obj} that reaches {v} m/s from rest after traveling {d} m, calculate the net force.",
        "The {obj} weighs {m} kg and accelerates from 0 to {v} m/s over {d} meters. Determine the applied force in N.",
        "Mass: {m} kg, Final velocity: {v} m/s, Distance: {d} m. Find the net force on this {obj} starting from rest.",
        "Consider a {m} kg {obj} that accelerates to {v} m/s over {d} meters from rest. What force was applied?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            v = np.random.randint(5, 40)
            d = np.random.randint(10, 100)
            m = np.random.randint(1, 20)
            obj = np.random.choice(objects)
            a = (v ** 2) / (2 * d)  # v^2 = 2ad
            expected_force = m * a
            
            prompt = "Question: " + prompt_format.format(obj=obj, m=m, v=v, d=d) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'm': m,
                'v': v,
                'd': d,
                'a': a,  # Hidden variable
                'expected_force': expected_force
            })
    
    return prompts_data


def gen_implicit_acceleration_spring_to_acceleration(samples_per_prompt):
    """
    Spring to Acceleration: "A spring with constant k compressed by x launches a ball of mass m. What is initial acceleration?"
    Bridge: k, x → F → a
    """
    prompts_data = []
    
    objects = ["ball", "mass", "object", "projectile", "block", "cart"]
    prompt_formats = [
        "A spring with constant {k} N/m is compressed by {x} meters. If it launches a {obj} of mass {m} kg, what is the initial acceleration in m/s²?",
        "Given a spring (k = {k} N/m) compressed {x} m that releases a {m} kg {obj}, calculate the initial acceleration.",
        "The spring has stiffness {k} N/m and is compressed {x} meters. It launches a {m} kg {obj}. Determine the initial acceleration.",
        "Spring constant: {k} N/m, Compression: {x} m, Mass: {m} kg. Find the initial acceleration of this {obj}.",
        "Consider a spring (k = {k} N/m) compressed {x} m launching a {m} kg {obj}. What is the initial acceleration?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            k = np.random.randint(10, 100)
            x = np.random.uniform(0.1, 2.0)
            m = np.random.randint(1, 10)
            obj = np.random.choice(objects)
            f = k * x  # F = kx (Hooke's law)
            expected_acceleration = f / m
            
            prompt = "Question: " + prompt_format.format(obj=obj, k=k, x=x, m=m) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'k': k,
                'x': x,
                'm': m,
                'f': f,  # Hidden variable (force)
                'a': expected_acceleration,  # Final hidden variable
                'expected_acceleration': expected_acceleration
            })
    
    return prompts_data


# ==========================================
# RADIUS AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_radius_from_area(samples_per_prompt):
    """
    Circle area → radius (hidden) → circumference
    Bridge: A → r → C
    """
    prompts_data = []
    
    shapes = ["circle", "circular disk", "round shape", "circular region", "disk", "circular area"]
    prompt_formats = [
        "A {shape} has an area of {area} square meters. What is its circumference in meters?",
        "Given a {shape} with a surface area of {area} square meters, calculate the perimeter.",
        "The {shape} covers {area} square meters of area. Determine the length around its edge in meters.",
        "Area: {area} square meters. Find the distance around the boundary of this {shape}.",
        "Consider a {shape} occupying {area} square meters. How many meters is the path around it?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            radius = np.random.randint(2, 20)
            shape = np.random.choice(shapes)
            area = np.pi * radius ** 2
            expected_circumference = 2 * np.pi * radius
            
            prompt = "Question: " + prompt_format.format(shape=shape, area=area) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'shape': shape,
                'area': area,
                'r': radius,  # Hidden variable
                'expected_circumference': expected_circumference
            })
    
    return prompts_data


def gen_implicit_radius_from_sphere_volume(samples_per_prompt):
    """
    Volume to Surface Area: "A sphere has volume V. What is its surface area?"
    Bridge: V → r → SA
    """
    prompts_data = []
    
    objects = ["sphere", "ball", "spherical object", "globe", "spherical body"]
    prompt_formats = [
        "A {obj} has a volume of {vol} cubic meters. What is its surface area in square meters?",
        "Given a {obj} with volume {vol} m³, calculate the surface area.",
        "The {obj} has a volume of {vol} cubic meters. Determine its surface area in m².",
        "Volume: {vol} m³. Find the surface area of this {obj}.",
        "Consider a {obj} with {vol} cubic meters of volume. What is its surface area?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            radius = np.random.uniform(1, 15)
            obj = np.random.choice(objects)
            volume = (4/3) * np.pi * (radius ** 3)
            expected_surface_area = 4 * np.pi * (radius ** 2)
            
            prompt = "Question: " + prompt_format.format(obj=obj, vol=volume) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'volume': volume,
                'r': radius,  # Hidden variable
                'expected_surface_area': expected_surface_area
            })
    
    return prompts_data


def gen_implicit_radius_from_cylinder_volume(samples_per_prompt):
    """
    Cylinder Volume to Diameter: "A cylinder with height h has volume V. What is its diameter?"
    Bridge: V, h → r → d
    """
    prompts_data = []
    
    objects = ["cylinder", "cylindrical tank", "tube", "pipe", "cylindrical container"]
    prompt_formats = [
        "A {obj} with height {h} cm has a volume of {vol} cubic centimeters. What is its diameter in cm?",
        "Given a {obj} of height {h} cm and volume {vol} cm³, calculate the diameter.",
        "The {obj} is {h} cm tall and has a volume of {vol} cubic cm. Determine its diameter.",
        "Height: {h} cm, Volume: {vol} cm³. Find the diameter of this {obj}.",
        "Consider a {obj} with height {h} cm and volume {vol} cm³. What is the diameter?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            radius = np.random.uniform(1, 15)
            h = np.random.uniform(5, 30)
            obj = np.random.choice(objects)
            volume = np.pi * (radius ** 2) * h
            expected_diameter = 2 * radius
            
            prompt = "Question: " + prompt_format.format(obj=obj, h=h, vol=volume) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'h': h,
                'volume': volume,
                'r': radius,  # Hidden variable
                'expected_diameter': expected_diameter
            })
    
    return prompts_data


# ==========================================
# SIDE LENGTH AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_side_length_from_volume(samples_per_prompt):
    """
    Cube volume → side length (hidden) → surface area
    Bridge: V → s → SA
    """
    prompts_data = []
    
    objects = ["cube", "cubic box", "cubic container", "cubic block", "box", "cubic structure"]
    prompt_formats = [
        "A {obj} has a volume of {vol} cubic centimeters. What is its total surface area in square centimeters?",
        "Given a {obj} that contains {vol} cubic centimeters of space, calculate the surface area.",
        "The {obj} has an internal capacity of {vol} cubic centimeters. Determine its exterior surface area.",
        "Volume: {vol} cubic centimeters. Find the total area of all faces of this {obj} in square centimeters.",
        "Consider a {obj} with {vol} cubic centimeters of volume. How much surface area does it have?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            side = np.random.randint(2, 15)
            obj = np.random.choice(objects)
            volume = side ** 3
            expected_surface_area = 6 * (side ** 2)
            
            prompt = "Question: " + prompt_format.format(obj=obj, vol=volume) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'volume': volume,
                's': side,  # Hidden variable
                'expected_surface_area': expected_surface_area
            })
    
    return prompts_data


def gen_implicit_side_length_from_area(samples_per_prompt):
    """
    Area to Diagonal: "A square has area A. What is the length of its diagonal?"
    Bridge: A → s → d
    """
    prompts_data = []
    
    objects = ["square", "square tile", "square plate", "square region", "quadrilateral"]
    prompt_formats = [
        "A {obj} has an area of {area} square centimeters. What is the length of its diagonal in cm?",
        "Given a {obj} with area {area} cm², calculate the diagonal length.",
        "The {obj} covers {area} square centimeters. Determine the diagonal length in cm.",
        "Area: {area} cm². Find the diagonal of this {obj}.",
        "Consider a {obj} with {area} cm² of area. What is its diagonal length?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            side = np.random.uniform(2, 20)
            obj = np.random.choice(objects)
            area = side ** 2
            expected_diagonal = side * np.sqrt(2)
            
            prompt = "Question: " + prompt_format.format(obj=obj, area=area) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'area': area,
                's': side,  # Hidden variable
                'expected_diagonal': expected_diagonal
            })
    
    return prompts_data


# ==========================================
# WAVELENGTH AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_wavelength_from_speed(samples_per_prompt):
    """
    Wave with speed and frequency → wavelength (hidden) → distance between n crests
    Bridge: v, f → λ → distance
    """
    prompts_data = []
    
    wave_types = ["wave", "sound wave", "water wave", "acoustic wave", "wave pattern", "oscillation"]
    prompt_formats = [
        "A {wave} travels at {speed} m/s with a frequency of {freq} Hz. What is the distance between {n} consecutive crests?",
        "Given a {wave} moving at {speed} m/s and oscillating at {freq} Hz, calculate the span between {n} adjacent wave peaks.",
        "The {wave} has a speed of {speed} m/s and frequency of {freq} Hz. Determine the separation of {n} successive crests in meters.",
        "Speed: {speed} m/s, Frequency: {freq} Hz. Find the distance covered by {n} complete wavelengths of this {wave}.",
        "Consider a {wave} at {freq} Hz traveling at {speed} m/s. How many meters separate the first crest from the {n}th crest?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            wavelength = np.random.randint(2, 20)
            freq = np.random.randint(2, 15)
            wave_type = np.random.choice(wave_types)
            speed = wavelength * freq  # v = λf
            n_crests = np.random.randint(3, 8)
            expected_distance = wavelength * (n_crests - 1)  # Distance between n crests
            
            prompt = "Question: " + prompt_format.format(wave=wave_type, speed=speed, freq=freq, n=n_crests) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'wave_type': wave_type,
                'speed': speed,
                'freq': freq,
                'n': n_crests,
                'wavelength': wavelength,  # Hidden variable
                'expected_distance': expected_distance
            })
    
    return prompts_data


# ==========================================
# CROSS-SECTIONAL AREA (via radius) AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_cross_section_from_flow(samples_per_prompt):
    """
    Water through cylindrical pipe with radius and speed → cross-sectional area (hidden) → volume
    Bridge: r, v → A → volume
    """
    prompts_data = []
    
    conduits = ["pipe", "cylindrical pipe", "tube", "conduit", "circular pipe", "cylindrical tube"]
    prompt_formats = [
        "Water flows through a {conduit} with a radius of {r} cm at a speed of {v} cm/s. How much water (in cubic cm) passes through after {t} seconds?",
        "Given a {conduit} with radius {r} cm where water moves at {v} cm/s, calculate the volume of water in cubic centimeters that flows through for {t} seconds.",
        "The {conduit} has a {r} cm radius and water travels at {v} cm/s. Determine the total volume discharged in {t} seconds.",
        "Radius: {r} cm, Flow speed: {v} cm/s. Find the cubic centimeters of water transported through this {conduit} after {t} seconds.",
        "Consider a {conduit} of {r} cm radius carrying water at {v} cm/s. What volume in cubic cm flows through in {t} seconds?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            radius = np.random.randint(2, 15)
            velocity = np.random.randint(5, 30)
            conduit = np.random.choice(conduits)
            time = np.random.randint(10, 60)
            area = np.pi * (radius ** 2)  # Hidden
            expected_volume = area * velocity * time
            
            prompt = "Question: " + prompt_format.format(conduit=conduit, r=radius, v=velocity, t=time) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'conduit': conduit,
                'r': radius,
                'v': velocity,
                't': time,
                'area': area,  # Hidden variable
                'expected_volume': expected_volume
            })
    
    return prompts_data


# ==========================================
# DISPLACEMENT AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_displacement_from_spring(samples_per_prompt):
    """
    Spring with spring constant and force → displacement (hidden) → potential energy
    Bridge: k, F → x → PE
    """
    prompts_data = []
    
    spring_types = ["spring", "elastic spring", "coil spring", "mechanical spring", "helical spring", "spring system"]
    prompt_formats = [
        "A {spring} with spring constant {k} N/m has {f} Newtons of force applied to it. What is the elastic potential energy stored in Joules?",
        "Given a {spring} with k = {k} N/m stretched by a force of {f} N, calculate the potential energy.",
        "The {spring} has stiffness {k} N/m and is under {f} N of force. Determine the stored elastic energy in Joules.",
        "Spring constant: {k} N/m, Applied force: {f} N. Find the potential energy stored in this {spring}.",
        "Consider a {spring} with spring constant {k} N/m pulled with {f} Newtons. How much elastic potential energy does it contain in Joules?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            displacement = np.random.randint(2, 15)
            k = np.random.randint(5, 30)
            spring = np.random.choice(spring_types)
            force = k * displacement  # F = kx
            expected_pe = 0.5 * k * (displacement ** 2)
            
            prompt = "Question: " + prompt_format.format(spring=spring, k=k, f=force) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'spring': spring,
                'k': k,
                'f': force,
                'x': displacement,  # Hidden variable
                'expected_pe': expected_pe
            })
    
    return prompts_data


# ==========================================
# MARKET CAP AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_market_cap_from_shares(samples_per_prompt):
    """
    Company with share price and shares → market cap (hidden) → P/E ratio
    Bridge: price, shares → market_cap → P/E
    """
    prompts_data = []
    
    company_types = ["company", "corporation", "firm", "business", "enterprise", "public company"]
    prompt_formats = [
        "A {company} has a share price of ${price} and {shares} million shares outstanding, with annual net income of ${income} million. What is its P/E ratio?",
        "Given a {company} trading at ${price} per share with {shares} million shares and ${income} million in net income, calculate the price-to-earnings ratio.",
        "The {company} has {shares} million shares at ${price} each and annual earnings of ${income} million. Determine the P/E ratio.",
        "Share price: ${price}, Shares outstanding: {shares} million, Net income: ${income} million. Find the P/E ratio for this {company}.",
        "Consider a {company} with {shares} million shares worth ${price} each that earns ${income} million annually. What is the price-to-earnings ratio?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            shares_millions = np.random.randint(50, 500)
            price = np.random.randint(20, 200)
            company = np.random.choice(company_types)
            market_cap = shares_millions * price
            pe_ratio = np.random.randint(10, 30)
            income = market_cap / pe_ratio
            
            prompt = "Question: " + prompt_format.format(company=company, price=price, shares=shares_millions, income=income) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'company': company,
                'price': price,
                'shares': shares_millions,
                'income': income,
                'market_cap': market_cap,  # Hidden variable
                'expected_pe': pe_ratio
            })
    
    return prompts_data


# ==========================================
# MOLES AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_moles_ideal_gas_to_mass(samples_per_prompt):
    """
    Ideal Gas to Mass: "A tank of volume V holds O2 at pressure P and temperature T. What is the mass?"
    Bridge: P, V, T → n → m
    """
    prompts_data = []
    
    gases = [
        ("Oxygen", "O2", 32.0),
        ("Nitrogen", "N2", 28.0),
        ("Helium", "He", 4.0),
        ("Carbon Dioxide", "CO2", 44.0),
        ("Argon", "Ar", 40.0)
    ]
    
    prompt_formats = [
        "A tank of volume {v} liters holds {gas_name} ({formula}) at pressure {p} atm and temperature {t} K. What is the mass of the gas in grams?",
        "Given a {v} L container of {gas_name} ({formula}) at {p} atm and {t} K, calculate the mass in grams.",
        "The tank contains {v} liters of {gas_name} ({formula}) at {p} atmospheres and {t} Kelvin. Determine the mass.",
        "Volume: {v} L, Gas: {gas_name} ({formula}), Pressure: {p} atm, Temperature: {t} K. Find the mass in grams.",
        "Consider {v} liters of {gas_name} ({formula}) at {p} atm and {t} K. What is the mass of the gas?"
    ]
    
    R = 0.0821  # L⋅atm/(mol⋅K)
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            gas_name, formula, molar_mass = gases[np.random.randint(0, len(gases))]
            v = np.random.uniform(1, 50)
            p = np.random.uniform(1, 10)
            t = np.random.uniform(273, 373)
            
            n = (p * v) / (R * t)  # PV = nRT
            expected_mass = n * molar_mass
            
            prompt = "Question: " + prompt_format.format(
                v=v, 
                gas_name=gas_name, 
                formula=formula, 
                p=p, 
                t=t
            ) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'gas_name': gas_name,
                'formula': formula,
                'molar_mass': molar_mass,
                'v': v,
                'p': p,
                't': t,
                'n': n,  # Hidden variable
                'expected_mass': expected_mass
            })
    
    return prompts_data


def gen_implicit_moles_mass_to_pressure(samples_per_prompt):
    """
    Mass to Pressure: "x grams of Helium in container V at temperature T. What is the pressure?"
    Bridge: mass → n → P
    """
    prompts_data = []
    
    gases = [
        ("Helium", "He", 4.0),
        ("Hydrogen", "H2", 2.0),
        ("Nitrogen", "N2", 28.0),
        ("Oxygen", "O2", 32.0)
    ]
    
    prompt_formats = [
        "{mass} grams of {gas_name} ({formula}) are placed in a container of volume {v} liters at temperature {t} K. What is the pressure in atm?",
        "Given {mass} g of {gas_name} ({formula}) in {v} L at {t} K, calculate the pressure.",
        "The container holds {mass} grams of {gas_name} ({formula}) with volume {v} liters and temperature {t} Kelvin. Determine the pressure.",
        "Mass: {mass} g, Gas: {gas_name} ({formula}), Volume: {v} L, Temperature: {t} K. Find the pressure in atmospheres.",
        "Consider {mass} grams of {gas_name} ({formula}) in {v} liters at {t} K. What is the pressure?"
    ]
    
    R = 0.0821  # L⋅atm/(mol⋅K)
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            gas_name, formula, molar_mass = gases[np.random.randint(0, len(gases))]
            mass = np.random.uniform(1, 100)
            v = np.random.uniform(1, 50)
            t = np.random.uniform(273, 373)
            
            n = mass / molar_mass  # Hidden
            expected_pressure = (n * R * t) / v  # PV = nRT
            
            prompt = "Question: " + prompt_format.format(
                mass=mass,
                gas_name=gas_name,
                formula=formula,
                v=v,
                t=t
            ) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'gas_name': gas_name,
                'formula': formula,
                'molar_mass': molar_mass,
                'mass': mass,
                'v': v,
                't': t,
                'n': n,  # Hidden variable
                'expected_pressure': expected_pressure
            })
    
    return prompts_data


# ==========================================
# ENERGY AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_energy_power_to_height(samples_per_prompt):
    """
    Power to Climb: "A motor uses P watts to lift mass m. How high can it lift the mass in time t?"
    Bridge: P, t → E → h
    """
    prompts_data = []
    
    objects = ["mass", "weight", "object", "load", "cargo", "block"]
    prompt_formats = [
        "A motor uses {p} watts to lift a {obj} of {m} kg. How high in meters can it lift the {obj} in {t} seconds?",
        "Given a motor rated at {p} W lifting a {m} kg {obj}, calculate the height reached in {t} seconds.",
        "The motor delivers {p} watts of power to lift a {m} kg {obj}. Determine the height in {t} seconds.",
        "Power: {p} W, Mass: {m} kg, Time: {t} s. Find the height this {obj} can be lifted.",
        "Consider a motor using {p} watts to lift a {m} kg {obj}. What height is reached after {t} seconds?"
    ]
    
    g = 9.8
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            p = np.random.randint(100, 2000)
            m = np.random.randint(5, 100)
            t = np.random.randint(5, 60)
            obj = np.random.choice(objects)
            
            energy = p * t  # E = Pt (Hidden)
            expected_height = energy / (m * g)  # h = E / (mg)
            
            prompt = "Question: " + prompt_format.format(p=p, m=m, obj=obj, t=t) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'p': p,
                'm': m,
                't': t,
                'g': g,
                'energy': energy,  # Hidden variable
                'expected_height': expected_height
            })
    
    return prompts_data


def gen_implicit_energy_friction_to_heat(samples_per_prompt):
    """
    Friction to Heat: "Block mass m slides distance d against friction f. If all energy goes into heat (specific heat c), 
    how much does temperature rise?"
    Bridge: f, d → W → ΔT
    """
    prompts_data = []
    
    objects = ["block", "object", "mass", "box", "sled"]
    materials = [
        ("aluminum", 900),
        ("copper", 385),
        ("iron", 450),
        ("steel", 470)
    ]
    
    prompt_formats = [
        "A {material} {obj} of mass {m} kg slides {d} meters against a friction force of {f} N. If all energy goes into heating the {obj} (specific heat {c} J/(kg⋅K)), how much does its temperature rise in Kelvin?",
        "Given a {m} kg {material} {obj} sliding {d} m with friction force {f} N, calculate the temperature rise. Specific heat: {c} J/(kg⋅K).",
        "The {material} {obj} ({m} kg) experiences {f} N of friction over {d} meters. With specific heat {c} J/(kg⋅K), determine the temperature increase.",
        "Mass: {m} kg, Material: {material}, Friction: {f} N, Distance: {d} m, Specific heat: {c} J/(kg⋅K). Find the temperature rise of this {obj}.",
        "Consider a {m} kg {material} {obj} sliding {d} m against {f} N friction. With c = {c} J/(kg⋅K), what is the temperature increase?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            material, c = materials[np.random.randint(0, len(materials))]
            m = np.random.uniform(0.5, 10)
            d = np.random.uniform(5, 50)
            f = np.random.uniform(5, 50)
            obj = np.random.choice(objects)
            
            work = f * d  # W = Fd (Hidden)
            expected_temp_rise = work / (m * c)  # Q = mcΔT
            
            prompt = "Question: " + prompt_format.format(
                material=material,
                obj=obj,
                m=m,
                d=d,
                f=f,
                c=c
            ) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'material': material,
                'c': c,
                'm': m,
                'd': d,
                'f': f,
                'work': work,  # Hidden variable (energy)
                'expected_temp_rise': expected_temp_rise
            })
    
    return prompts_data


# ==========================================
# FORCE AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_force_pressure_to_acceleration(samples_per_prompt):
    """
    Pressure to Acceleration: "A piston of area A is pushed by pressure P. If it moves mass m, what is acceleration?"
    Bridge: P, A → F → a
    """
    prompts_data = []
    
    objects = ["piston", "plunger", "cylinder", "ram"]
    prompt_formats = [
        "A {obj} with cross-sectional area {a} cm² is pushed by a pressure of {p} Pa. If it moves a mass of {m} kg, what is the acceleration in m/s²?",
        "Given a {obj} of area {a} cm² under pressure {p} Pa pushing mass {m} kg, calculate the acceleration.",
        "The {obj} has area {a} cm² and experiences pressure {p} Pa. It accelerates a {m} kg mass. Determine the acceleration.",
        "Area: {a} cm², Pressure: {p} Pa, Mass: {m} kg. Find the acceleration produced by this {obj}.",
        "Consider a {obj} with {a} cm² area at {p} Pa moving {m} kg. What is the acceleration?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            area_cm2 = np.random.uniform(10, 200)
            area_m2 = area_cm2 / 10000  # Convert to m²
            pressure = np.random.uniform(10000, 200000)  # Pa
            m = np.random.uniform(1, 20)
            obj = np.random.choice(objects)
            
            force = pressure * area_m2  # F = PA (Hidden)
            expected_acceleration = force / m
            
            prompt = "Question: " + prompt_format.format(
                obj=obj,
                a=area_cm2,
                p=pressure,
                m=m
            ) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'area_cm2': area_cm2,
                'area_m2': area_m2,
                'pressure': pressure,
                'm': m,
                'force': force,  # Hidden variable
                'expected_acceleration': expected_acceleration
            })
    
    return prompts_data


def gen_implicit_force_electric_field_to_acceleration(samples_per_prompt):
    """
    Electric Field to Acceleration: "A particle of charge q and mass m is in electric field E. What is acceleration?"
    Bridge: q, E → F → a
    """
    prompts_data = []
    
    particles = [
        ("electron", -1.6e-19, 9.11e-31),
        ("proton", 1.6e-19, 1.67e-27),
        ("alpha particle", 3.2e-19, 6.64e-27)
    ]
    
    prompt_formats = [
        "A {particle} (charge {q_str} C, mass {m_str} kg) is placed in an electric field of {e} N/C. What is its acceleration in m/s²?",
        "Given a {particle} with q = {q_str} C and m = {m_str} kg in field E = {e} N/C, calculate the acceleration.",
        "The {particle} has charge {q_str} C, mass {m_str} kg, and is in an {e} N/C electric field. Determine the acceleration.",
        "Charge: {q_str} C, Mass: {m_str} kg, Field: {e} N/C. Find the acceleration of this {particle}.",
        "Consider a {particle} (q = {q_str} C, m = {m_str} kg) in E = {e} N/C. What is the acceleration?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            particle_name, q, m = particles[np.random.randint(0, len(particles))]
            e = np.random.uniform(1000, 100000)
            
            force = q * e  # F = qE (Hidden)
            expected_acceleration = force / m
            
            prompt = "Question: " + prompt_format.format(
                particle=particle_name,
                q_str=q,
                m_str=m,
                e=e
            ) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'particle': particle_name,
                'q': q,
                'm': m,
                'e': e,
                'force': force,  # Hidden variable
                'expected_acceleration': expected_acceleration
            })
    
    return prompts_data


# ==========================================
# FREQUENCY/PERIOD AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_frequency_pendulum_to_gravity(samples_per_prompt):
    """
    Pendulum to Gravity: "A pendulum of length L has period T. What is gravitational acceleration g?"
    Bridge: L, T → f → g
    """
    prompts_data = []
    
    objects = ["pendulum", "simple pendulum", "bob", "swing"]
    prompt_formats = [
        "A {obj} with length {l} meters has a period of {t} seconds. What is the gravitational acceleration g in m/s²?",
        "Given a {obj} of length {l} m oscillating with period {t} s, calculate the gravitational acceleration.",
        "The {obj} is {l} meters long and completes one swing in {t} seconds. Determine g in m/s².",
        "Length: {l} m, Period: {t} s. Find the gravitational acceleration for this {obj}.",
        "Consider a {obj} with {l} m length and {t} s period. What is g?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            l = np.random.uniform(0.5, 3.0)
            g_actual = np.random.uniform(9.0, 10.5)  # Allow for different planets
            obj = np.random.choice(objects)
            
            # T = 2π√(L/g), so g = 4π²L/T²
            t = 2 * np.pi * np.sqrt(l / g_actual)  # Calculate period from g
            frequency = 1 / t  # Hidden
            
            expected_g = (4 * np.pi**2 * l) / (t ** 2)
            
            prompt = "Question: " + prompt_format.format(
                obj=obj,
                l=l,
                t=t
            ) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'l': l,
                't': t,
                'f': frequency,  # Hidden variable
                'expected_g': expected_g
            })
    
    return prompts_data


# ==========================================
# ANGLE (TRIGONOMETRY) AS HIDDEN VARIABLE
# ==========================================

def gen_implicit_angle_ramp_to_acceleration(samples_per_prompt):
    """
    Ramp to Acceleration: "A block slides down a frictionless ramp of height h and length L. What is acceleration?"
    Bridge: h, L → θ → a
    """
    prompts_data = []
    
    objects = ["block", "object", "mass", "box", "cart", "body"]
    prompt_formats = [
        "A {obj} slides down a frictionless ramp with height {h} meters and length {l} meters. What is its acceleration down the ramp in m/s²?",
        "Given a {obj} on a frictionless ramp (height {h} m, length {l} m), calculate the acceleration.",
        "The {obj} is on a ramp that rises {h} meters over {l} meters of length. Determine the acceleration down the slope.",
        "Height: {h} m, Length: {l} m. Find the acceleration of this {obj} sliding down the frictionless ramp.",
        "Consider a {obj} on a frictionless ramp with {h} m height and {l} m length. What is the acceleration?"
    ]
    
    g = 9.8
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            theta_deg = np.random.uniform(10, 60)
            theta_rad = np.radians(theta_deg)
            l = np.random.uniform(2, 20)
            h = l * np.sin(theta_rad)
            obj = np.random.choice(objects)
            
            sin_theta = h / l  # Hidden trig calculation
            expected_acceleration = g * sin_theta
            
            prompt = "Question: " + prompt_format.format(
                obj=obj,
                h=h,
                l=l
            ) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'h': h,
                'l': l,
                'theta_deg': theta_deg,
                'theta_rad': theta_rad,
                'sin_theta': sin_theta,  # Hidden variable
                'g': g,
                'expected_acceleration': expected_acceleration
            })
    
    return prompts_data


def gen_implicit_angle_components_to_magnitude(samples_per_prompt):
    """
    Components to Magnitude: "A force has x-component Fx and y-component Fy. What acceleration does it give mass m?"
    Bridge: Fx, Fy → F_total → a
    """
    prompts_data = []
    
    prompt_formats = [
        "A force vector has x-component {fx} N and y-component {fy} N. What acceleration does it give a mass of {m} kg in m/s²?",
        "Given a force with Fx = {fx} N and Fy = {fy} N acting on {m} kg, calculate the magnitude of acceleration.",
        "The force has components {fx} N (x-direction) and {fy} N (y-direction). Determine the acceleration of a {m} kg mass.",
        "Fx: {fx} N, Fy: {fy} N, Mass: {m} kg. Find the magnitude of acceleration.",
        "Consider a force with components {fx} N and {fy} N on a {m} kg object. What is the acceleration magnitude?"
    ]
    
    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            fx = np.random.uniform(-50, 50)
            fy = np.random.uniform(-50, 50)
            m = np.random.uniform(1, 20)
            
            f_total = np.sqrt(fx**2 + fy**2)  # Hidden
            theta = np.arctan2(fy, fx)  # Also hidden
            expected_acceleration = f_total / m
            
            prompt = "Question: " + prompt_format.format(
                fx=fx,
                fy=fy,
                m=m
            ) + " Answer (step-by-step): "
            
            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'fx': fx,
                'fy': fy,
                'm': m,
                'f_total': f_total,  # Hidden variable
                'theta': theta,  # Also hidden
                'expected_acceleration': expected_acceleration
            })
    
    return prompts_data


def gen_implicit_velocity_from_distance_time(samples_per_prompt):
    """Distance and time imply velocity, which is then used to extrapolate a different segment of travel."""
    prompts_data = []

    objects = ["car", "train", "runner", "bike", "plane", "boat"]
    prompt_formats = [
        "A {obj} covers {d} meters in {t} seconds. If it keeps the same pace, how long will it take to cover another {d2} meters?",
        "Given that a {obj} goes {d} m in {t} s, estimate the time needed to travel an additional {d2} meters.",
        "The {obj} moves {d} meters in {t} seconds. How many seconds are required for it to go {d2} more meters at the same speed?",
        "Distance: {d} m, Time: {t} s. At this pace, what is the travel time for a further {d2} m for this {obj}?",
        "Consider a {obj} that covers {d} meters in {t} seconds. How long does it need to travel {d2} extra meters?"
    ]

    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            d = np.random.randint(20, 200)
            t = np.random.randint(2, 20)
            d2 = np.random.randint(20, 250)
            obj = np.random.choice(objects)

            v = d / t
            expected_time = d2 / v

            prompt = "Question: " + prompt_format.format(obj=obj, d=d, t=t, d2=d2) + " Answer (step-by-step): "

            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'd': d,
                't': t,
                'd2': d2,
                'v': v,
                'expected_time': expected_time,
            })

    return prompts_data


def gen_implicit_current_from_charge_time(samples_per_prompt):
    """Charge and time imply current, which is then used to extrapolate a later charge transfer."""
    prompts_data = []

    objects = ["device", "wire", "resistor", "circuit", "appliance"]
    prompt_formats = [
        "A {obj} transfers {q} coulombs in {t} seconds. If the current stays constant, how much charge moves in the next {t2} seconds?",
        "Given a {obj} carrying {q} C over {t} s, estimate the charge moved during an additional {t2} seconds.",
        "The {obj} moves {q} coulombs in {t} seconds. How much charge passes through it over another {t2} seconds?",
        "Charge: {q} C, Time: {t} s. At the same rate, find the charge flow in the next {t2} seconds for this {obj}.",
        "Consider a {obj} with {q} coulombs over {t} seconds. How much charge will flow in a further {t2} seconds?"
    ]

    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            q = np.random.randint(10, 100)
            t = np.random.randint(2, 20)
            t2 = np.random.randint(5, 30)
            obj = np.random.choice(objects)

            i = q / t
            expected_charge = i * t2

            prompt = "Question: " + prompt_format.format(obj=obj, q=q, t=t, t2=t2) + " Answer (step-by-step): "

            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'q': q,
                't': t,
                't2': t2,
                'i': i,
                'expected_charge': expected_charge,
            })

    return prompts_data


def gen_implicit_force_from_mass_acceleration(samples_per_prompt):
    """Mass and acceleration imply force, which is then used to compute impulse over time."""
    prompts_data = []

    objects = ["cart", "box", "sled", "probe", "vehicle"]
    prompt_formats = [
        "A {obj} of mass {m} kg accelerates at {a} m/s² for {t} seconds. How much momentum change does it accumulate?",
        "Given a {m} kg {obj} with acceleration {a} m/s² over {t} s, determine the total impulse delivered.",
        "The {obj} has mass {m} kg and acceleration {a} m/s². What impulse is delivered after {t} seconds?",
        "Mass: {m} kg, Acceleration: {a} m/s². Find the impulse gained by this {obj} over {t} seconds.",
        "Consider a {m} kg {obj} accelerating at {a} m/s². Calculate the impulse after {t} seconds."
    ]

    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            m = np.random.randint(2, 20)
            a = np.random.randint(1, 10)
            t = np.random.randint(2, 20)
            obj = np.random.choice(objects)

            force = m * a
            expected_impulse = force * t

            prompt = "Question: " + prompt_format.format(obj=obj, m=m, a=a, t=t) + " Answer (step-by-step): "

            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'm': m,
                'a': a,
                't': t,
                'force': force,
                'expected_impulse': expected_impulse,
            })

    return prompts_data


def gen_implicit_acceleration_from_velocity_change(samples_per_prompt):
    """Velocity change over time implies acceleration, which is then used to compute force."""
    prompts_data = []

    objects = ["car", "train", "drone", "runner", "cart"]
    prompt_formats = [
        "A {obj} changes speed from {u} m/s to {v} m/s in {t} seconds. What acceleration did it experience?",
        "Given a {m} kg {obj} going from {u} m/s to {v} m/s in {t} s, determine its acceleration.",
        "The {obj} speeds up from {u} m/s to {v} m/s over {t} seconds. What is the acceleration of the motion?",
        "Initial speed: {u} m/s, Final speed: {v} m/s, Time: {t} s. Find the acceleration of this {obj}.",
        "Consider a {m} kg {obj} whose speed changes from {u} m/s to {v} m/s in {t} seconds. Calculate the acceleration."
    ]

    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            u = np.random.randint(0, 20)
            v = np.random.randint(u + 1, u + 20)
            t = np.random.randint(2, 15)
            m = np.random.randint(2, 20)
            obj = np.random.choice(objects)

            a = (v - u) / t
            expected_force = m * a

            prompt = "Question: " + prompt_format.format(obj=obj, u=u, v=v, t=t, m=m) + " Answer (step-by-step): "

            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'u': u,
                'v': v,
                't': t,
                'm': m,
                'a': a,
                'expected_force': expected_force,
            })

    return prompts_data


def gen_implicit_radius_from_circumference(samples_per_prompt):
    """Circumference implies radius, which is then used to compute area."""
    prompts_data = []

    prompt_formats = [
        "A circular track has circumference {c} meters. What area would a filled disk with that same boundary cover?",
        "Given a circle with circumference {c} m, calculate the area enclosed by the same ring.",
        "The circumference of a round table is {c} meters. Determine the tabletop area.",
        "Circumference: {c} m. Find the area enclosed by the circle.",
        "Consider a circle with circumference {c} meters. What is the area inside it?"
    ]

    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            r = np.random.randint(2, 30)
            c = 2 * np.pi * r
            area = np.pi * r ** 2

            prompt = "Question: " + prompt_format.format(c=f"{c:.2f}") + " Answer (step-by-step): "

            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'c': c,
                'r': r,
                'expected_area': area,
            })

    return prompts_data


def gen_implicit_side_length_from_perimeter(samples_per_prompt):
    """Perimeter implies side length, which is then used to compute area."""
    prompts_data = []

    prompt_formats = [
        "A square fence has perimeter {p} meters. What area of ground does it enclose?",
        "Given a square with perimeter {p} m, determine the land area inside it.",
        "The square's perimeter is {p} meters. Calculate the enclosed area.",
        "Perimeter: {p} m. Find the area inside the square.",
        "Consider a square with perimeter {p} meters. How many square meters are inside?"
    ]

    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            s = np.random.randint(2, 30)
            p = 4 * s
            expected_area = s ** 2

            prompt = "Question: " + prompt_format.format(p=p) + " Answer (step-by-step): "

            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'p': p,
                's': s,
                'expected_area': expected_area,
            })

    return prompts_data


def gen_implicit_frequency_from_period(samples_per_prompt):
    """Period implies frequency, which is then used to count oscillations over time."""
    prompts_data = []

    objects = ["pendulum", "oscillator", "wave", "signal", "vibration"]
    prompt_formats = [
        "A metronome ticks once every {period} seconds. How many ticks will it make in {t} seconds?",
        "Given a {obj} period of {period} s, calculate how many cycles occur in {t} seconds.",
        "The {obj} repeats every {period} seconds. Determine the number of repetitions over {t} seconds.",
        "Period: {period} s. Find the total number of oscillations in {t} seconds for this {obj}.",
        "Consider a {obj} with period {period} seconds. How many complete cycles happen in {t} seconds?"
    ]

    for format_id, prompt_format in enumerate(prompt_formats):
        for _ in range(samples_per_prompt):
            period = np.random.randint(2, 10)
            t = np.random.randint(10, 50)
            obj = np.random.choice(objects)

            frequency = 1 / period
            expected_cycles = frequency * t

            prompt = "Question: " + prompt_format.format(obj=obj, period=period, t=t) + " Answer (step-by-step): "

            prompts_data.append({
                'prompt': prompt,
                'format_id': format_id,
                'object': obj,
                'period': period,
                't': t,
                'frequency': frequency,
                'expected_cycles': expected_cycles,
            })

    return prompts_data


# ==========================================
# UTILITY FUNCTIONS
# ==========================================

def get_all_generators():
    """
    Returns a dictionary mapping experiment names to their generator functions.
    """
    return {
        # Velocity as hidden variable
        'velocity_from_ke': gen_implicit_velocity_from_ke,
        'velocity_from_ke_uniform_t': gen_implicit_velocity_from_ke_uniform_t,
        'velocity_from_ke_momentum': gen_implicit_velocity_from_ke_momentum,
        'velocity_from_momentum': gen_implicit_velocity_from_momentum,
        'velocity_from_energy_conservation': gen_implicit_velocity_from_energy_conservation,
        'velocity_from_distance_time': gen_implicit_velocity_from_distance_time,
        
        # Current as hidden variable
        'current_from_power': gen_implicit_current_from_power,
        'current_from_voltage': gen_implicit_current_from_voltage,
        'current_from_charge_time': gen_implicit_current_from_charge_time,
        
        # Acceleration as hidden variable
        'acceleration_force_to_distance': gen_implicit_acceleration_force_to_distance,
        'acceleration_velocity_to_force': gen_implicit_acceleration_velocity_to_force,
        'acceleration_spring': gen_implicit_acceleration_spring_to_acceleration,
        'acceleration_from_velocity_change': gen_implicit_acceleration_from_velocity_change,
        
        # Radius as hidden variable
        'radius_from_area': gen_implicit_radius_from_area,
        'radius_from_sphere_volume': gen_implicit_radius_from_sphere_volume,
        'radius_from_cylinder_volume': gen_implicit_radius_from_cylinder_volume,
        'radius_from_circumference': gen_implicit_radius_from_circumference,
        
        # Side length as hidden variable
        'side_length_from_volume': gen_implicit_side_length_from_volume,
        'side_length_from_area': gen_implicit_side_length_from_area,
        'side_length_from_perimeter': gen_implicit_side_length_from_perimeter,
        
        # Wavelength as hidden variable
        'wavelength_from_speed': gen_implicit_wavelength_from_speed,
        
        # Cross-sectional area as hidden variable
        'cross_section_from_flow': gen_implicit_cross_section_from_flow,
        
        # Displacement as hidden variable
        'displacement_from_spring': gen_implicit_displacement_from_spring,
        
        # Market cap as hidden variable
        'market_cap_from_shares': gen_implicit_market_cap_from_shares,
        
        # Moles as hidden variable
        'moles_ideal_gas_to_mass': gen_implicit_moles_ideal_gas_to_mass,
        'moles_mass_to_pressure': gen_implicit_moles_mass_to_pressure,
        
        # Energy as hidden variable
        'energy_power_to_height': gen_implicit_energy_power_to_height,
        'energy_friction_to_heat': gen_implicit_energy_friction_to_heat,
        
        # Force as hidden variable
        'force_pressure_to_acceleration': gen_implicit_force_pressure_to_acceleration,
        'force_electric_field': gen_implicit_force_electric_field_to_acceleration,
        'force_from_mass_acceleration': gen_implicit_force_from_mass_acceleration,
        
        # Frequency as hidden variable
        'frequency_pendulum_to_gravity': gen_implicit_frequency_pendulum_to_gravity,
        'frequency_from_period': gen_implicit_frequency_from_period,
        
        # Angle as hidden variable
        'angle_ramp_to_acceleration': gen_implicit_angle_ramp_to_acceleration,
        'angle_components_to_magnitude': gen_implicit_angle_components_to_magnitude,
    }


def generate_prompts_for_experiment(experiment_name, samples_per_format=50):
    """
    Generate prompts for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment (key from get_all_generators())
        samples_per_format: Number of samples to generate per template format
    
    Returns:
        List of prompt dictionaries
    """
    generators = get_all_generators()
    if experiment_name not in generators:
        raise ValueError(f"Unknown experiment: {experiment_name}. Available: {list(generators.keys())}")
    
    generator_func = generators[experiment_name]
    return generator_func(samples_per_format)
