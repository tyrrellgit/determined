"""
Manim scene for the two-mass-spring-damper system.

This scene runs the same simulation as the Python example but renders a
textbook-style diagram: two masses as boxes and springs as zig-zag lines.
It uses the Rust `PyKalmanFilter` (PyO3) to compute the filter estimates.

Run (recommended, low-quality preview):
  pip install manim  # community edition
  python -m manim -pql examples/manim_two_mass_spring_damper.py TwoMassSpringDamperScene

For higher quality or to save a video/GIF, adjust manim flags (e.g. -pqm, -pqh)
or use `--renderer=opengl` if installed and desired.

"""

import numpy as np
from manim import (
    Scene, Axes, VMobject, DashedVMobject,
    Line, Rectangle, Text, ValueTracker,
    always_redraw, linear,
    LEFT, UP, DOWN,
    BLUE, GREEN, GREY
)

# Use the Rust PyO3 wrapper for the Kalman filter
from determined import PyKalmanFilter as PyKalman


def setup_two_mass_spring_damper_params():
    m = 0.1
    m_large = 1.0
    k = 0.091
    b = 0.0036

    a = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-k / m, -b / m, k / m, b / m],
        [0.0, 0.0, 0.0, 1.0],
        [k / m_large, b / m_large, -k / m_large, -b / m_large],
    ], dtype=float)

    dt = 0.1
    I = np.eye(4)
    f = I + dt * a

    h = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ], dtype=float)

    sigma_q = np.diag([0.3 * dt, 0.05 * dt, 0.3 * dt, 0.05 * dt])
    q = sigma_q @ sigma_q.T
    sigma_r = np.diag([0.1, 0.1])
    r = sigma_r @ sigma_r.T

    x0 = np.array([-1.0, 0.0, -0.5, 0.0], dtype=float)

    return f, h, q, r, x0, dt


def simulate_rust(num_steps=100, process_noise=0.0, measurement_noise=0.05):
    """Run the rust-backed Kalman filter and return trajectories.
    Returns (traj_true, traj_est) arrays of shape (num_steps, 4).
    """
    f, h, q, r, x0, dt = setup_two_mass_spring_damper_params()

    rust_filter = PyKalman(state_dim=f.shape[0], meas_dim=h.shape[0])
    rust_filter.set_state_transition(f.flatten().tolist())
    rust_filter.set_measurement_matrix(h.tolist())
    rust_filter.set_process_noise(q.tolist())
    rust_filter.set_measurement_noise(r.tolist())
    rust_filter.set_state(x0.tolist())
    rust_filter.set_covariance(2.0)

    x_true = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    traj_true = np.zeros((num_steps, 4))
    traj_est = np.zeros((num_steps, 4))

    proc_noise = np.zeros(f.shape[0])
    meas_noise = np.zeros(h.shape[0])

    for step in range(num_steps):
        rust_filter.predict()
        if process_noise > 0.0:
            proc_noise = np.random.normal(0.0, measurement_noise, f.shape[0])
        if measurement_noise > 0.0:
            meas_noise = np.random.normal(0.0, measurement_noise, h.shape[0])
        x_true = f @ x_true + proc_noise
        z = (h @ x_true + meas_noise).tolist()
        state_vec = np.array(rust_filter.update(z))

        traj_true[step] = x_true
        traj_est[step] = state_vec

    return traj_true, traj_est


def make_damper_path(p1: np.ndarray, p2: np.ndarray, width=0.3):
    """Return points forming a damper (rectangle) between p1 and p2.
    p1/p2 are 2D points. A damper is drawn as a small box perpendicular to the line.
    """
    p1 = np.array([p1[0], p1[1]])
    p2 = np.array([p2[0], p2[1]])
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < 1e-8:
        return None
    
    tangent = v / length
    normal = np.array([-tangent[1], tangent[0]])
    
    # Center of damper
    center = (p1 + p2) / 2
    
    # Damper box corners (rectangle perpendicular to spring line)
    half_width = width / 2
    box_length = 0.25
    
    c1 = center - tangent * box_length / 2 + normal * half_width
    c2 = center - tangent * box_length / 2 - normal * half_width
    c3 = center + tangent * box_length / 2 - normal * half_width
    c4 = center + tangent * box_length / 2 + normal * half_width
    
    # Return as polygon corners
    return [c1.tolist(), c2.tolist(), c3.tolist(), c4.tolist(), c1.tolist()]


def make_spring_path(p1: np.ndarray, p2: np.ndarray, coils=8, amplitude=0.15):
    """Return a list of points forming a zig-zag spring from p1 to p2.
    p1/p2 are numpy arrays (x, y, z) or 2D points.
    """
    p1 = np.array([p1[0], p1[1]])
    p2 = np.array([p2[0], p2[1]])
    v = p2 - p1
    length = np.linalg.norm(v)
    if length < 1e-8:
        return [p1.tolist(), p2.tolist()]

    tangent = v / length
    normal = np.array([-tangent[1], tangent[0]])

    # Coil points along the line
    segments = coils * 2
    points = [p1.tolist()]
    for i in range(1, segments):
        t = i / segments
        base = p1 + t * v
        # alternate offset
        sign = 1.0 if i % 2 == 0 else -1.0
        offset = normal * amplitude * sign
        points.append((base + offset).tolist())
    points.append(p2.tolist())
    return points


class TwoMassSpringDamperScene(Scene):
    """Manim Scene that animates the two-mass-spring-damper system.
    
    Shows two rows: top row is TRUE system, bottom row is FILTER ESTIMATE.
    Left wall fixed, then spring-mass1-spring-mass2 configuration.
    """

    def construct(self):
        # Simulation
        num_steps = 400
        run_time = 20.0  # seconds
        traj_true, traj_est = simulate_rust(
            num_steps=num_steps, 
            process_noise=0.05,
            measurement_noise=0.01)

        scale = 1.0
        offset_y_true = 1.0  # True system at y=1
        offset_y_est = -1.0  # Estimate at y=-1

        # Ground/wall line across both rows
        ground_line = Line(LEFT * 4 + UP * 1.3, LEFT * 4 + DOWN * 1.3, color=GREY, stroke_width=4)
        self.add(ground_line)

        # Add hatching to indicate fixed wall
        wall_hatch_spacing = 0.3
        for i in np.arange(-1.3, 1.4, wall_hatch_spacing):
            hatch = Line(LEFT * 3.95 + UP * i, LEFT * 3.8 + UP * (i - 0.2), color=GREY, stroke_width=2)
            self.add(hatch)

        # Mass rectangles (true = filled, est = outline)
        mass_width = 0.7
        mass_height = 0.5
        
        # TRUE SYSTEM
        mass_M_true = Rectangle(width=mass_width * 1.2, height=mass_height, fill_color=BLUE, 
                                fill_opacity=0.8, stroke_color=BLUE, stroke_width=2)
        mass_m_true = Rectangle(width=mass_width, height=mass_height, fill_color=GREEN, 
                                fill_opacity=0.8, stroke_color=GREEN, stroke_width=2)
        
        # ESTIMATED SYSTEM (outline only)
        mass_M_est = Rectangle(width=mass_width * 1.2, height=mass_height, fill_opacity=0.0, 
                               stroke_color=BLUE, stroke_width=2)
        mass_m_est = Rectangle(width=mass_width, height=mass_height, fill_opacity=0.0, 
                               stroke_color=GREEN, stroke_width=2)

        # Add labels to masses
        label_M_true = Text("M", font_size=20, color=GREY).move_to(mass_M_true.get_center())
        label_m_true = Text("m", font_size=20, color=GREY).move_to(mass_m_true.get_center())
        label_M_est = Text("M", font_size=20, color=BLUE).move_to(mass_M_est.get_center())
        label_m_est = Text("m", font_size=20, color=GREEN).move_to(mass_m_est.get_center())

        self.add(mass_M_true, mass_m_true, label_M_true, label_m_true)
        self.add(mass_M_est, mass_m_est, label_M_est, label_m_est)

        # Fixed wall anchor points
        wall_true = np.array([-3.5, offset_y_true, 0.0])
        wall_est = np.array([-3.5, offset_y_est, 0.0])

        # Function to draw spring between two objects
        def spring_between_mobjects(m1, m2, coils=8, amplitude=0.12, stroke_color=GREY):
            p1 = m1.get_center()
            p2 = m2.get_center()
            pts = make_spring_path(np.array(p1)[:2], np.array(p2)[:2], coils=coils, amplitude=amplitude)
            return VMobject().set_points_as_corners([np.array([x, y, 0.0]) for x, y in pts]).set_stroke(
                color=stroke_color, width=2)

        # Function to draw damper between two objects
        def damper_between_mobjects(m1, m2, stroke_color=GREY):
            p1 = m1.get_center()
            p2 = m2.get_center()
            pts = make_damper_path(np.array(p1)[:2], np.array(p2)[:2], width=0.25)
            if pts is None:
                return VMobject()
            return VMobject().set_points_as_corners([np.array([x, y, 0.0]) for x, y in pts]).set_stroke(
                color=stroke_color, width=2).set_fill(color=stroke_color, opacity=0.3)

        # Dynamic springs (M-m only) for both true and estimated
        spring_M_m_true = always_redraw(
            lambda: spring_between_mobjects(mass_M_true, mass_m_true, 
                                           coils=6, amplitude=0.10, stroke_color=BLUE))
        
        # Damper between M and m (parallel to spring)
        damper_M_m_true = always_redraw(
            lambda: damper_between_mobjects(mass_M_true, mass_m_true, stroke_color=GREY))
        
        spring_M_m_est = always_redraw(
            lambda: spring_between_mobjects(mass_M_est, mass_m_est, 
                                           coils=6, amplitude=0.10, stroke_color=GREEN))
        
        # Damper between M and m for estimated
        damper_M_m_est = always_redraw(
            lambda: damper_between_mobjects(mass_M_est, mass_m_est, stroke_color=GREY))

        self.add(spring_M_m_true, damper_M_m_true, spring_M_m_est, damper_M_m_est)

        # Labels for rows
        label_true = Text("TRUE", font_size=24, color=BLUE).move_to(np.array([-5.2, offset_y_true, 0.0]))
        label_est = Text("ESTIMATE", font_size=24, color=GREEN).move_to(np.array([-5.5, offset_y_est, 0.0]))
        self.add(label_true, label_est)

        # Main title
        title = Text("Two-Mass-Spring-Damper Kalman Filter", font_size=32, color=GREY).to_edge(UP)
        self.add(title)

        # --- Axes plots: positions and errors over time ---
        # Determine plotting ranges
        num_steps = traj_true.shape[0]
        dt = 0.1
        t_max = (num_steps - 1) * dt

        pos_min = float(min(traj_true[:,0].min(), traj_true[:,2].min(), traj_est[:,0].min(), traj_est[:,2].min()))
        pos_max = float(max(traj_true[:,0].max(), traj_true[:,2].max(), traj_est[:,0].max(), traj_est[:,2].max()))
        y_margin = max(0.1, 0.1 * (pos_max - pos_min))
        pos_ymin = pos_min - y_margin
        pos_ymax = pos_max + y_margin

        # Errors
        errs_M = np.abs(traj_true[:,0] - traj_est[:,0])
        errs_m = np.abs(traj_true[:,2] - traj_est[:,2])
        err_max = float(max(errs_M.max(), errs_m.max(), 1e-3))
        # Create three axes: positions, error_M, error_m (wider and spaced)
        # Wider axes and extra spacing for readability
        axes_pos = Axes(
            x_range=[0, t_max, max(1.0, t_max/4)],
            y_range=[pos_ymin, pos_ymax, (pos_ymax-pos_ymin)/4],
            x_length=3.8, y_length=1.6, tips=False, stroke_width=1.5, color=GREY
        ).move_to(np.array([-5.0, -3.0, 0.0]))
        axes_err_M = Axes(
            x_range=[0, t_max, max(1.0, t_max/4)],
            y_range=[0, err_max, max(err_max/4, 0.01)],
            x_length=3.8, y_length=1.6, tips=False, stroke_width=1.5, color=GREY
        ).move_to(np.array([0.0, -3.0, 0.0]))
        axes_err_m = Axes(
            x_range=[0, t_max, max(1.0, t_max/4)],
            y_range=[0, err_max, max(err_max/4, 0.01)],
            x_length=3.8, y_length=1.6, tips=False, stroke_width=1.5, color=GREY
        ).move_to(np.array([5.0, -3.0, 0.0]))

        # Add numeric tick labels (manual, compatible with this Manim version)
        # X ticks (shared across plots) — render under each axes so labels line up
        x_ticks = np.linspace(0.0, t_max, num=5)
        for xt in x_ticks:
            for ax in (axes_pos, axes_err_M, axes_err_m):
                lbl = Text(f"{xt:.2f}", font_size=9, color=GREY)
                # place just below the axis bottom for each axes
                bottom_y = ax.y_range[0] if hasattr(ax, 'y_range') else pos_ymin
                lbl.move_to(ax.coords_to_point(xt, bottom_y))
                lbl.shift(DOWN * 0.14)
                self.add(lbl)

        # Y ticks for position axes
        y_ticks = np.linspace(pos_ymin, pos_ymax, num=4)
        for yt in y_ticks:
            lbl = Text(f"{yt:.2f}", font_size=7, color=GREY)
            lbl.move_to(axes_pos.coords_to_point(0.0, yt))
            lbl.shift(LEFT * 0.14)
            self.add(lbl)

        # Y ticks for error axes (both error plots)
        err_ticks = np.linspace(0.0, err_max, num=4)
        for et in err_ticks:
            lblM = Text(f"{et:.2f}", font_size=7, color=GREY)
            lblM.move_to(axes_err_M.coords_to_point(0.0, et))
            lblM.shift(LEFT * 0.14)
            self.add(lblM)
            lblm = Text(f"{et:.2f}", font_size=7, color=GREY)
            lblm.move_to(axes_err_m.coords_to_point(0.0, et))
            lblm.shift(LEFT * 0.14)
            self.add(lblm)

        # Add axis labels with units
        pos_label = Text("Position (m)", font_size=10, color=GREY).next_to(axes_pos, UP, buff=0.1)
        err_M_label = Text("Error (M) (m)", font_size=10, color=GREY).next_to(axes_err_M, UP, buff=0.1)
        err_m_label = Text("Error (m) (m)", font_size=10, color=GREY).next_to(axes_err_m, UP, buff=0.1)

        # Add y-axis labels (units on left side)
        pos_y_label = Text("m", font_size=8, color=GREY).next_to(axes_pos, LEFT, buff=0.1)
        err_M_y_label = Text("m", font_size=8, color=GREY).next_to(axes_err_M, LEFT, buff=0.1)
        err_m_y_label = Text("m", font_size=8, color=GREY).next_to(axes_err_m, LEFT, buff=0.1)

        self.add(axes_pos, axes_err_M, axes_err_m, pos_label, err_M_label, err_m_label,
                 pos_y_label, err_M_y_label, err_m_y_label)

        # Value tracker must be defined before any updaters/curves that reference it
        t = ValueTracker(0.0)

        # Helper to build VMobject from series up to current time
        def series_to_curve(axes, series, step_idx):
            # step_idx is a step index (0 to num_steps-1), convert to time
            tval = step_idx * dt
            idx = min(int(step_idx), len(series)-1)
            pts = []
            for i in range(idx+1):
                x = i * dt
                y = float(series[i])
                pts.append(axes.coords_to_point(x, y))
            # add interpolated last point if between steps
            frac = step_idx - idx
            if frac > 1e-6 and idx < len(series)-1:
                y_interp = (1-frac)*series[idx] + frac*series[idx+1]
                pts.append(axes.coords_to_point((idx+1)*dt, float(y_interp)))
            if len(pts) < 2:
                return VMobject()
            return VMobject().set_points_as_corners(pts).set_stroke(width=2)

        # Create always_redraw curves (ValueTracker `t` is already defined)
        # Position curves: true solid
        curve_M_true = always_redraw(
            lambda: series_to_curve(axes_pos, traj_true[:,0], t.get_value()).set_stroke(color=BLUE, width=3, opacity=0.4)
            )
        curve_m_true = always_redraw(
            lambda: series_to_curve(axes_pos, traj_true[:,2], t.get_value()).set_stroke(color=GREEN, width=3, opacity=0.4),
            )

        # Estimated curves: draw as dashed overlays using DashedVMobject
        def maked_dashed(axes, traj, color):
            base = series_to_curve(axes, traj, t.get_value())
            if getattr(base, 'points', None) is None or len(base.points) == 0:
                return VMobject()
            base.set_stroke(color=color, width=3, opacity=1.0)
            dashed = DashedVMobject(base, num_dashes=36)
            return dashed

        curve_M_est = always_redraw(
            lambda: maked_dashed(axes_pos, traj_est[:,0], BLUE)
            )
        curve_m_est = always_redraw(
            lambda: maked_dashed(axes_pos, traj_est[:,2], GREEN)
            )

        # Error curves: separate axes for each mass
        curve_err_M = always_redraw(
            lambda: series_to_curve(axes_err_M, errs_M, t.get_value()).set_stroke(color=BLUE, width=3)
            )
        curve_err_m = always_redraw(
            lambda: series_to_curve(axes_err_m, errs_m, t.get_value()).set_stroke(color=GREEN, width=3)
            )

        self.add(
            curve_M_true,
            curve_m_true,
            curve_M_est,
            curve_m_est,
            curve_err_M,
            curve_err_m
        )

        # Value tracker to animate through discrete trajectory steps
        t = ValueTracker(0.0)

        def get_interp_positions(vtracker):
            """Interpolate between discrete simulation steps."""
            val = np.clip(vtracker.get_value(), 0, num_steps - 1)
            i0 = int(np.floor(val))
            i1 = min(i0 + 1, num_steps - 1)
            alpha = val - i0
            
            # True positions: x1=mass M, x3=mass m
            x1_t = (1 - alpha) * traj_true[i0, 0] + alpha * traj_true[i1, 0]
            x3_t = (1 - alpha) * traj_true[i0, 2] + alpha * traj_true[i1, 2]
            
            # Estimated positions
            x1_e = (1 - alpha) * traj_est[i0, 0] + alpha * traj_est[i1, 0]
            x3_e = (1 - alpha) * traj_est[i0, 2] + alpha * traj_est[i1, 2]
            
            return x1_t * scale, x3_t * scale, x1_e * scale, x3_e * scale

        # Updaters to move masses
        def update_mass_M_true(mob):
            x1_t, _, _, _ = get_interp_positions(t)
            mob.move_to(np.array([x1_t, offset_y_true, 0.0]))
            label_M_true.move_to(mob.get_center())

        def update_mass_m_true(mob):
            _, x3_t, _, _ = get_interp_positions(t)
            mob.move_to(np.array([x3_t, offset_y_true, 0.0]))
            label_m_true.move_to(mob.get_center())

        def update_mass_M_est(mob):
            _, _, x1_e, _ = get_interp_positions(t)
            mob.move_to(np.array([x1_e, offset_y_est, 0.0]))
            label_M_est.move_to(mob.get_center())

        def update_mass_m_est(mob):
            _, _, _, x3_e = get_interp_positions(t)
            mob.move_to(np.array([x3_e, offset_y_est, 0.0]))
            label_m_est.move_to(mob.get_center())

        mass_M_true.add_updater(update_mass_M_true)
        mass_m_true.add_updater(update_mass_m_true)
        mass_M_est.add_updater(update_mass_M_est)
        mass_m_est.add_updater(update_mass_m_est)

        # Play the animation
        self.play(t.animate.set_value(num_steps - 1), run_time=run_time, rate_func=linear)

        # Hold final frame
        self.wait(1)


# If run directly by manim, it will pick up TwoMassSpringDamperScene
if __name__ == '__main__':
    # For quick local run without manim CLI (useful for debugging), render a single frame:
    traj_true, traj_est = simulate_rust(num_steps=10)
    print('Simulated', traj_true.shape)
