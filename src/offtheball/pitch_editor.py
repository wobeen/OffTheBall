"""Select the visible playing surface; no metre coordinates are needed."""
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk


class PitchEditor:
    def __init__(self, parent, pixels, on_saved):
        self.window = tk.Toplevel(parent)
        self.window.title("그라운드 영역 · 경기장 밖 사람 제외")
        self.window.configure(bg="#141e2c")
        self.points = []
        self.pixels = pixels
        self.on_saved = on_saved
        h, w = pixels.shape[:2]
        self.scale = min(1000 / w, 620 / h, 1)
        tk.Label(self.window, text="화면에 보이는 그라운드 가장자리를 순서대로 클릭하세요. 좌표 입력은 필요 없습니다.",
                 bg="#141e2c", fg="white", pady=12).pack()
        self.canvas = tk.Canvas(self.window, width=round(w*self.scale), height=round(h*self.scale), highlightthickness=0)
        self.canvas.pack(padx=12)
        self.image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).resize((round(w*self.scale),round(h*self.scale))))
        self.canvas.create_image(0, 0, image=self.image, anchor="nw")
        self.canvas.bind("<Button-1>", self.click)
        bar = tk.Frame(self.window, bg="#141e2c")
        bar.pack(pady=12)
        ttk.Button(bar, text="마지막 점 취소", command=self.undo).pack(side="left", padx=8)
        ttk.Button(bar, text="영역 적용", command=self.save).pack(side="left", padx=8)

    def click(self, event):
        h, w = self.pixels.shape[:2]
        self.points.append([min(w-1, max(0, event.x/self.scale)), min(h-1, max(0, event.y/self.scale))])
        self.draw()

    def draw(self):
        self.canvas.delete("boundary")
        coords = [(x*self.scale,y*self.scale) for x,y in self.points]
        if len(coords)>2:
            self.canvas.create_polygon(coords, fill="", outline="#69e2b3", width=3, tags="boundary")
        for i,(x,y) in enumerate(coords):
            self.canvas.create_oval(x-4,y-4,x+4,y+4,fill="#69e2b3",tags="boundary")
            self.canvas.create_text(x+12,y-10,text=str(i+1),fill="white",tags="boundary")

    def undo(self):
        if self.points:
            self.points.pop()
            self.draw()

    def save(self):
        polygon = np.asarray(self.points, np.float32)
        if len(polygon)<3 or not cv2.isContourConvex(polygon) or cv2.contourArea(polygon)<100:
            messagebox.showinfo("영역 확인", "서로 교차하지 않도록 경기장 가장자리를 따라 3개 이상의 점을 지정하세요.", parent=self.window)
            return
        self.on_saved(polygon)
        self.window.destroy()
