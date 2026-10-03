import queue
import tkinter as tk

from amanuensis.asr.streaming import Update


class Overlay:
    """Always-on-top window: committed text in white, tentative text in grey.

    `push` is thread-safe; `run` must be called from the main thread.
    """

    def __init__(self, banner: str = "") -> None:
        self._q: queue.Queue[Update] = queue.Queue()
        self._committed = ""
        self._root = tk.Tk()
        self._root.title("Amanuensis")
        self._root.attributes("-topmost", True)
        self._root.configure(bg="#1e1e1e")
        self._root.geometry("760x140+100+100")
        self._banner = tk.Label(self._root, text=banner, fg="#e5a50a", bg="#1e1e1e", anchor="w")
        self._banner.pack(fill="x")
        self._text = tk.Text(self._root, bg="#1e1e1e", wrap="word", font=("Segoe UI", 16), borderwidth=0)
        self._text.tag_config("committed", foreground="#ffffff")
        self._text.tag_config("tentative", foreground="#8a8a8a")
        self._text.pack(fill="both", expand=True, padx=8, pady=4)

    def push(self, update: Update) -> None:
        self._q.put(update)

    def run(self) -> None:
        self._poll()
        self._root.mainloop()

    def close(self) -> None:
        self._root.after(0, self._root.destroy)

    def _poll(self) -> None:
        tentative = None
        while True:
            try:
                u = self._q.get_nowait()
            except queue.Empty:
                break
            if u.committed:
                self._committed = (self._committed + " " + u.committed).strip()[-400:]
            tentative = u.tentative
            if u.final:
                self._committed += "\n"
        if tentative is not None:
            self._text.delete("1.0", "end")
            self._text.insert("end", self._committed + " ", "committed")
            self._text.insert("end", tentative, "tentative")
            self._text.see("end")
        self._root.after(50, self._poll)
