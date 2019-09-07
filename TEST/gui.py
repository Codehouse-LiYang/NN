# -*- coding:utf-8 -*-
import tkinter


"事件"
def incident(button):
    button.widget["background"] = "Azure"  # active鼠标按下触发
    button.widget["text"] = "OK"  # 鼠标接触触发


"窗体设置"
MainForm = tkinter.Tk()
MainForm.title("三酷猫")  # 窗体标题属性
MainForm["background"] = "#FF0000"
MainForm.geometry("250x150")  # geometry  几何学
MainForm.iconbitmap("A:/TEST/bing.ico")  # *.ico  window图标文件
"按钮"
button = tkinter.Button(MainForm, cnf={"text": "退出", "fg": "Black", "bg": "Coral"})  # 设置按钮属性
button.bind("<Button-1>", incident)  # 将incident绑定到button上  bind  捆绑
button.pack(side="bottom", anchor="e")  # 将button放到MainForm上(底部，右对齐)
"启动窗体运行"
MainForm.mainloop()  # mainloop  主循环




