"""
Interactive point-cloud viewer with a live brightness slider.

Standalone -- works on any Open3D-readable .ply, not tied to the
reconstruction pipeline. Point colors are baked in at reconstruction time
straight from Mini-3D-Recon's DarkIR-enhanced input frame (no manual color
grading), so real footage can render darker than a viewer wants; this lets
brightness be adjusted live after the fact, on the CPU-cheap rendering side,
without re-running any model or touching the saved .ply file.

Uses Open3D's GUI/rendering API (not the plain o3d.visualization.
draw_geometries() every other preview in this project uses) specifically
because draw_geometries() has no way to attach custom UI controls like a
slider -- it's a fixed, non-extensible window.

Usage: python -m src.fusion.view_point_cloud --ply path/to/cloud.ply
"""

import argparse

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


class BrightnessViewer:
    def __init__(self, pcd: o3d.geometry.PointCloud, window_title: str):
        # Kept separate from pcd.colors (which gets overwritten on every
        # slider move) so brightness scaling always starts from the
        # original values -- otherwise repeated small adjustments would
        # compound/clip irreversibly instead of being a clean function of
        # the current slider position.
        self.original_colors = np.asarray(pcd.colors).copy()
        self.pcd = pcd

        self.window = gui.Application.instance.create_window(window_title, 1280, 800)
        em = self.window.theme.font_size

        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.05, 0.05, 0.05, 1.0])

        self.material = rendering.MaterialRecord()
        self.material.shader = "defaultUnlit"  # point colors are pre-baked -- no scene lighting to fight
        self.material.point_size = 2.0
        self.scene_widget.scene.add_geometry("pcd", self.pcd, self.material)

        bounds = self.pcd.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60, bounds, bounds.get_center())

        panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        panel.add_child(gui.Label("Brightness"))
        self.slider = gui.Slider(gui.Slider.DOUBLE)
        self.slider.set_limits(0.1, 4.0)
        self.slider.double_value = 1.0
        self.slider.set_on_value_changed(self._on_brightness_changed)
        panel.add_child(self.slider)
        self.value_label = gui.Label("1.00x")
        panel.add_child(self.value_label)

        panel.add_child(gui.Label("Point size"))
        self.size_slider = gui.Slider(gui.Slider.DOUBLE)
        self.size_slider.set_limits(1.0, 8.0)
        self.size_slider.double_value = 2.0
        self.size_slider.set_on_value_changed(self._on_point_size_changed)
        panel.add_child(self.size_slider)

        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self.scene_widget)
        self.window.add_child(panel)
        self.panel = panel

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        panel_width = 18 * self.window.theme.font_size
        self.panel.frame = gui.Rect(r.get_right() - panel_width, r.y, panel_width, r.height)
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_width, r.height)

    def _on_brightness_changed(self, value):
        colors = np.clip(self.original_colors * value, 0.0, 1.0)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        # Open3DScene has no in-place "update colors" call -- cheapest
        # correct way to push a color change to the renderer is remove+re-add
        # the same named geometry, which is fine at interactive-slider cost
        # for a single point cloud (no per-frame animation loop involved).
        self.scene_widget.scene.remove_geometry("pcd")
        self.scene_widget.scene.add_geometry("pcd", self.pcd, self.material)
        self.value_label.text = f"{value:.2f}x"

    def _on_point_size_changed(self, value):
        self.material.point_size = value
        self.scene_widget.scene.remove_geometry("pcd")
        self.scene_widget.scene.add_geometry("pcd", self.pcd, self.material)


def main():
    parser = argparse.ArgumentParser(description="View a point cloud with a live brightness slider")
    parser.add_argument("--ply", required=True)
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.ply)
    print(f"loaded {len(pcd.points)} points from {args.ply}")
    if len(pcd.points) == 0:
        raise ValueError(f"{args.ply} contains no points")

    gui.Application.instance.initialize()
    BrightnessViewer(pcd, window_title=f"Brightness viewer -- {args.ply}")
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
