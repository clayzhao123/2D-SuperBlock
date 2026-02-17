from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

from .buffer import ReplayBuffer, Transition
from .env import SuperblockEnv
from .forage_train import run as run_forage
from .models import ForwardModel
from .monitor import save_checkpoint, write_dashboard, write_history_csv
from .train import action_to_onehot, parse_visible_cells, train_night
from .utils import exp, set_seed


@dataclass
class UIConfig:
    visible_cells: list[tuple[int, int]]
    superblock_count: int
    init_points: list[tuple[int, int]]
    days: int
    steps_per_day: int
    mode: str
    motion_checkpoint_path: str


def parse_init_points(spec: str) -> list[tuple[int, int]]:
    pts = parse_visible_cells(spec)
    if len(pts) != 4:
        raise ValueError("superblock 初始位置必须包含 4 个点，格式如 2:2,3:2,3:3,2:3")
    return pts


def center_of(points: list[tuple[int, int]]) -> tuple[float, float]:
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    return sx / len(points), sy / len(points)


def run_ui() -> None:
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import messagebox

    class App:
        def __init__(self, root: tk.Tk) -> None:
            self.root = root
            self.root.title("Superblock v1.2 Pixel Monitor")
            self.root.configure(bg="#0A0A0A")

            self.colors = {
                "bg": "#0A0A0A",
                "header": "#0D0D0D",
                "surface": "#111111",
                "surface_border": "#1E1E1E",
                "panel_border": "#2A2A2A",
                "primary": "#FF6633",
                "secondary": "#FF9900",
                "text": "#FFFFFF",
                "text_secondary": "#888888",
                "grid_line": "#1A1A1A",
                "grid_bg": "#080808",
                "input_bg": "#0D0D0D",
                "input_border": "#333333",
            }

            self.pixel_font = self._pick_font(("Press Start 2P", "Courier New", "Courier"), size=8)
            self.pixel_title_font = self._pick_font(("Press Start 2P", "Courier New", "Courier"), size=9)
            self.header_font = self._pick_font(("Press Start 2P", "Courier New", "Courier"), size=10)
            self.mono_font = self._pick_font(("JetBrains Mono", "Courier New", "Courier"), size=11)
            self.metric_font = self._pick_font(("JetBrains Mono", "Courier New", "Courier"), size=12)

            self.cell = 12
            self.grid_cells = 41
            self.max_traj_days = 5

            header = tk.Frame(root, bg=self.colors["header"], height=40, highlightthickness=1, highlightbackground=self.colors["surface_border"])
            header.grid(row=0, column=0, columnspan=2, sticky="ew")
            header.grid_propagate(False)
            tk.Label(
                header,
                text="SUPERBLOCK SIMULATION",
                bg=self.colors["header"],
                fg=self.colors["text"],
                font=self.header_font,
            ).grid(row=0, column=0, padx=12, pady=12, sticky="w")

            self.env_canvas = tk.Canvas(
                root,
                width=self.grid_cells * self.cell,
                height=self.grid_cells * self.cell,
                bg=self.colors["grid_bg"],
                highlightthickness=1,
                highlightbackground=self.colors["surface_border"],
            )
            self.env_canvas.grid(row=1, column=0, padx=12, pady=12, sticky="n")

            panel = tk.Frame(
                root,
                bg=self.colors["surface"],
                padx=16,
                pady=16,
                highlightthickness=1,
                highlightbackground=self.colors["panel_border"],
            )
            panel.grid(row=1, column=1, padx=10, pady=12, sticky="n")

            self.visible_cells_var = tk.StringVar(value="19:19")
            self.superblock_count_var = tk.StringVar(value="1")
            self.init_points_var = tk.StringVar(value="2:2,3:2,3:3,2:3")
            self.days_var = tk.StringVar(value="100")
            self.steps_var = tk.StringVar(value="100")
            self.mode_var = tk.StringVar(value="motion")
            self.motion_ckpt_var = tk.StringVar(value="artifacts/train.ckpt")

            self._pixel_label(panel, "调整面板", 0, 0, bold=True)
            self._pixel_label(panel, "训练模式(motion/forage)", 1, 0)
            self._styled_entry(panel, self.mode_var, 1)
            self._pixel_label(panel, "可视单元格坐标", 2, 0)
            self._styled_entry(panel, self.visible_cells_var, 2)
            self._pixel_label(panel, "superblock数量(v1.3预留)", 3, 0)
            self._styled_entry(panel, self.superblock_count_var, 3)
            self._pixel_label(panel, "superblock初始位置(4点)", 4, 0)
            self._styled_entry(panel, self.init_points_var, 4)
            self._pixel_label(panel, "训练天数", 5, 0)
            self._styled_entry(panel, self.days_var, 5)
            self._pixel_label(panel, "每天采样步数", 6, 0)
            self._styled_entry(panel, self.steps_var, 6)
            self._pixel_label(panel, "觅食模式运动ckpt", 7, 0)
            self._styled_entry(panel, self.motion_ckpt_var, 7)

            self.run_btn = tk.Button(
                panel,
                text="RUN",
                command=self.on_run,
                bg=self.colors["primary"],
                fg="#000000",
                activebackground=self.colors["secondary"],
                activeforeground="#000000",
                width=16,
                height=2,
                relief="flat",
                bd=0,
                font=self.pixel_font,
                cursor="hand2",
            )
            self.run_btn.grid(row=8, column=0, columnspan=2, pady=10)
            self.run_btn.bind("<Enter>", lambda _e: self.run_btn.configure(bg=self.colors["secondary"]))
            self.run_btn.bind("<Leave>", lambda _e: self.run_btn.configure(bg=self.colors["primary"]))

            self._pixel_label(panel, "训练指标", 9, 0, bold=True)
            self.metrics_var = tk.StringVar(value="day=0 | score=0.000 | mse=0.000000")
            tk.Label(
                panel,
                textvariable=self.metrics_var,
                bg=self.colors["surface"],
                fg=self.colors["secondary"],
                font=self.metric_font,
            ).grid(row=10, column=0, columnspan=2, sticky="w")

            self._pixel_label(panel, "信任图（可视单元格命中次数）", 11, 0, bold=True)
            self.trust_canvas = tk.Canvas(
                panel,
                width=360,
                height=180,
                bg=self.colors["grid_bg"],
                highlightthickness=1,
                highlightbackground=self.colors["surface_border"],
            )
            self.trust_canvas.grid(row=12, column=0, columnspan=2, pady=6)

            self.history: list[dict[str, float]] = []
            self.day_paths: list[tuple[int, list[tuple[float, float]]]] = []

            self.draw_base_grid([(19, 19)], [(2, 2), (3, 2), (3, 3), (2, 3)])

        def _pick_font(self, candidates: tuple[str, ...], size: int, weight: str = "normal") -> tuple[str, int, str]:
            available = set(tkfont.families())
            family = next((name for name in candidates if name in available), candidates[-1])
            return (family, size, weight)

        def _styled_entry(self, parent: tk.Widget, var: tk.StringVar, row: int) -> tk.Entry:
            entry = tk.Entry(
                parent,
                textvariable=var,
                width=28,
                bg=self.colors["input_bg"],
                fg=self.colors["secondary"],
                insertbackground=self.colors["secondary"],
                relief="flat",
                highlightthickness=1,
                highlightbackground=self.colors["input_border"],
                highlightcolor=self.colors["primary"],
                bd=0,
                font=self.mono_font,
            )
            entry.grid(row=row, column=1, pady=2)
            return entry

        def _pixel_label(self, parent: tk.Widget, text: str, row: int, col: int, *, bold: bool = False) -> None:
            tk.Label(
                parent,
                text=text.upper(),
                bg=self.colors["surface"],
                fg=self.colors["primary"] if bold else self.colors["text_secondary"],
                font=self.pixel_title_font if bold else self.pixel_font,
            ).grid(row=row, column=col, sticky="w", pady=2)

        def config_from_ui(self) -> UIConfig:
            visible_cells = parse_visible_cells(self.visible_cells_var.get())
            init_points = parse_init_points(self.init_points_var.get())
            superblock_count = int(self.superblock_count_var.get())
            days = int(self.days_var.get())
            steps = int(self.steps_var.get())
            mode = self.mode_var.get().strip().lower()
            if mode not in {"motion", "forage"}:
                raise ValueError("训练模式必须是 motion 或 forage")
            if superblock_count < 1:
                raise ValueError("superblock数量必须>=1")
            return UIConfig(
                visible_cells=visible_cells,
                superblock_count=superblock_count,
                init_points=init_points,
                days=days,
                steps_per_day=steps,
                mode=mode,
                motion_checkpoint_path=self.motion_ckpt_var.get().strip(),
            )

        def draw_base_grid(self, visible_cells: list[tuple[int, int]], points: list[tuple[int, int]]) -> None:
            self.env_canvas.delete("all")
            for x in range(self.grid_cells):
                for y in range(self.grid_cells):
                    color = self.colors["grid_bg"]
                    if (x, y) in visible_cells:
                        color = self.colors["grid_line"]
                    self.draw_cell(x, y, color)
            for x, y in points:
                self.draw_cell(x, y, self.colors["primary"])
            self.draw_scanlines(self.env_canvas, self.grid_cells * self.cell, self.grid_cells * self.cell)

        def draw_cell(self, x: int, y: int, color: str) -> None:
            x0 = x * self.cell
            y0 = (self.grid_cells - 1 - y) * self.cell
            x1 = x0 + self.cell
            y1 = y0 + self.cell
            self.env_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=self.colors["grid_line"])

        def draw_scanlines(self, canvas: tk.Canvas, width: int, height: int) -> None:
            for y in range(0, height, 4):
                canvas.create_line(0, y, width, y, fill="#000000", width=1, stipple="gray50")

        def draw_trajectories(self, visible_cells: list[tuple[int, int]], points: list[tuple[int, int]]) -> None:
            self.draw_base_grid(visible_cells, points)
            shades = ["#1A1A1A", "#66321E", "#99431F", "#CC5429", "#FF6633"]
            recent = self.day_paths[-self.max_traj_days :]
            for idx, (_day, path) in enumerate(recent):
                shade = shades[min(idx, len(shades) - 1)]
                for cx, cy in path:
                    self.draw_cell(int(round(cx)), int(round(cy)), shade)

            for x, y in points:
                self.draw_cell(x, y, self.colors["primary"])
            self.draw_scanlines(self.env_canvas, self.grid_cells * self.cell, self.grid_cells * self.cell)

        def draw_trust_graph(self) -> None:
            self.trust_canvas.delete("all")
            w, h = 360, 180
            self.trust_canvas.create_rectangle(0, 0, w, h, fill=self.colors["grid_bg"], outline=self.colors["surface_border"])
            values = [row.get("visible_hits", 0.0) for row in self.history]
            if not values:
                return
            lo, hi = 0.0, max(values)
            span = hi - lo if hi != lo else 1.0
            pts = []
            for i, v in enumerate(values):
                x = 10 + (i / max(1, len(values) - 1)) * (w - 20)
                y = h - 10 - ((v - lo) / span) * (h - 20)
                pts.extend([x, y])
            self.trust_canvas.create_line(*pts, fill=self.colors["secondary"], width=2)
            self.draw_scanlines(self.trust_canvas, w, h)

        def on_run(self) -> None:
            try:
                cfg = self.config_from_ui()
            except Exception as exc:
                messagebox.showerror("配置错误", str(exc))
                return

            ok = messagebox.askyesno("确认训练", "是否执行训练？")
            if not ok:
                return

            self.run_btn.configure(state="disabled")
            try:
                self.run_training(cfg)
                messagebox.showinfo("完成", "训练已完成。")
            except Exception as exc:
                messagebox.showerror("训练失败", str(exc))
            finally:
                self.run_btn.configure(state="normal")

        def run_training(self, cfg: UIConfig) -> None:
            if cfg.mode == "forage":
                run_forage(
                    argparse.Namespace(
                        days=cfg.days,
                        steps_per_day=cfg.steps_per_day,
                        food_count=3,
                        hunger_interval=25,
                        hunger_death_steps=75,
                        vision_radius=3,
                        seed=42,
                        motion_checkpoint_path=cfg.motion_checkpoint_path,
                        motion_output_checkpoint="artifacts/train_forage_tuned.ckpt",
                        motion_epochs_per_day=10,
                        motion_batch_size=64,
                        motion_lr=1e-2,
                        motion_score_k=50.0,
                        motion_score_w_mse=0.5,
                        motion_score_w_exact=0.2,
                        motion_score_w_near=0.3,
                        out_checkpoint="artifacts/forage.ckpt",
                        out_metrics_csv="artifacts/forage_metrics.csv",
                        out_dashboard="artifacts/forage_dashboard.html",
                    )
                )
                self.metrics_var.set("forage done | dashboard=artifacts/forage_dashboard.html")
                return

            set_seed(42)
            rng = random.Random(42)
            env = SuperblockEnv(init_points=cfg.init_points, visible_cells=cfg.visible_cells)
            model = ForwardModel(seed=42)
            buffer = ReplayBuffer()
            self.history = []
            self.day_paths = []

            for day_idx in range(1, cfg.days + 1):
                invalid_count = 0
                visible_hits = 0
                day_path: list[tuple[float, float]] = []

                state_t, img_t = env.reset()
                day_path.append(center_of(env.points))
                for _ in range(cfg.steps_per_day):
                    action = env.sample_action(rng)
                    state_tp1, img_tp1, invalid = env.step(action)
                    if invalid:
                        invalid_count += 1
                    if any(px > 0.0 for row in img_tp1 for px in row):
                        visible_hits += 1
                    buffer.add(
                        Transition(
                            state_t=state_t.copy(),
                            action_vec=action_to_onehot(action),
                            invalid_move=invalid,
                            img_t=[row[:] for row in img_t],
                            state_tp1=state_tp1.copy(),
                            img_tp1=[row[:] for row in img_tp1],
                        )
                    )
                    state_t, img_t = state_tp1, img_tp1
                    day_path.append(center_of(env.points))

                pos_mse, exact_acc, near_acc = train_night(
                    model,
                    buffer,
                    batch_size=64,
                    lr=1e-2,
                    epochs=30,
                    seed=42 + day_idx,
                )
                score = 0.5 * exp(-50.0 * pos_mse) + 0.2 * exact_acc + 0.3 * near_acc
                row = {
                    "day_idx": float(day_idx),
                    "invalid_move_ratio": invalid_count / float(cfg.steps_per_day),
                    "pos_mse": pos_mse,
                    "exact_acc": exact_acc,
                    "near_acc": near_acc,
                    "score": score,
                    "visible_hits": float(visible_hits),
                }
                self.history.append(row)
                self.day_paths.append((day_idx, day_path))

                self.metrics_var.set(f"day={day_idx} | score={score:.3f} | mse={pos_mse:.6f}")
                self.draw_trajectories(cfg.visible_cells, env.points)
                self.draw_trust_graph()
                self.root.update_idletasks()
                self.root.update()

                save_checkpoint(
                    "artifacts/ui_train.ckpt",
                    day_idx=day_idx,
                    model=model,
                    buffer=buffer,
                    history=self.history,
                    visible_cells=cfg.visible_cells,
                )
                write_dashboard("artifacts/ui_dashboard.html", self.history, cfg.visible_cells)
                write_history_csv("artifacts/ui_metrics.csv", self.history)

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    run_ui()
