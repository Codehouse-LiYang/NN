# -*- coding:utf-8 -*-
import sys
import tkinter


"事件"
def incident(button):
    button.widget["background"] = "Azure"  # active鼠标按下触发
    button.widget["text"] = "OK"  # 鼠标接触触发
    sys.exit()


"窗体设置"
MainForm = tkinter.Tk()
MainForm.title("三酷猫")  # 窗体标题属性
MainForm["background"] = "#FF0000"
MainForm.geometry("700x600")  # geometry  几何学
MainForm.iconbitmap("A:/TEST/bing.ico")  # *.ico  window图标文件
"==================================================================================================Button按钮组件"
button = tkinter.Button(MainForm, cnf={"text": "退出", "fg": "Black", "bg": "Coral"})  # 设置按钮属性
button.bind("<Button-1>", incident)  # 将incident绑定到button上  bind  捆绑
button.pack(side="bottom", anchor="e")  # 将button放到MainForm上(底部，右对齐)
# print(help(button))  # 查看组件属性
"===================================================================================================Lable标签组件"
lable_1 = tkinter.Label(MainForm, text="三酷猫:")  # 标题
# photo = tkinter.PhotoImage(file="A:/TEST/woveC.gif")  # 只接受gif文件
# lable_2 = tkinter.Label(MainForm, image=photo)
lable_1.pack(side="left")
# lable_2.pack(side="left")
"===================================================================================================Entry单行文本"
entry = tkinter.Entry(MainForm, width=10)  # 单行文本输入
entry.pack(side="right")
"===================================================================================================Text多行文本"
text = tkinter.Text(MainForm, width=10, height=3)
text.pack(side="bottom")
"===================================================================================================Checkbutton复选框组件"
var = tkinter.StringVar()  # 字符串变量子类实例化
checkbutton = tkinter.Checkbutton(MainForm, cnf={"text": "蓝猫", "variable": var, "onvalue": "RGB", "offvalue": "L", "fg": "blue"})  # 创建带蓝色标题复选框
checkbutton.pack(side="top")
"===================================================================================================Radiobutton单选框组件"
v = tkinter.IntVar()
radiobutton = tkinter.Radiobutton(MainForm, cnf={"text": "ONE", "variable": v, "value": 1})
radiobutton.pack(anchor="w")
"启动窗体运行"
MainForm.mainloop()  # mainloop  主循环




