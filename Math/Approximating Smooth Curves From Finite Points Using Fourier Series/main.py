import subprocess
from matplotlib import pyplot as plt
from matplotlib import animation as animation
from math import ceil


print("Enter the points on the curve :", "RULES : ", "1.Give points serially as complex numbers.", "2.If it's a closed curve you need to input the first point at least twice (once at the beginning and once at the end).", "Example : 1+2j, 3+4j, 5+7j, 0+0j.... don't leave space before or after +", sep='\n')
argv1 = input("Insert Numbers Here : ")
points0 = argv1.split(",")
x0 = [complex(i).real for i in points0 if i]
y0 = [complex(i).imag for i in points0 if i]
frametime = 20  #in ms
flag = True
while flag:
    print("", "Enter the max value of abs(n)", "This kinda like a smoothness factor. Too small value makes circle like curve and too large makes pointy curve, so do some experiment and find the optimal value", "This is due to the fact that I use linear approximations", sep='\n')
    argv2 = input("Enter Number Here : ")
    print("", "Enter time for animation : ", "Too large value may cause overflow, in that case change change the c program or the frametime.", sep='\n')
    time = float(input("Enter Number Here : "))
    argv3 = str(ceil(time/frametime*1e3))
    print("", "Make sure you have a compiled c program named a.out for calcuations", "", sep='\n')

    output = subprocess.run(["./a.out", argv1, argv2, argv3], capture_output=True, check=True, text=True)
    print("Doing some calculations. Please wait.....")

    temp_var1 = output.stdout.split(";")
    # print(output.stdout)
    # print(argv3, argv2, argv1)
    points = temp_var1[0].split("\n")
    x = [complex(i).real for i in points if i]
    y = [complex(i).imag for i in points if i]

    plt.plot(x0, y0, "r.")
    plt.plot(x, y, label="Preview")
    plt.gca().set_aspect('equal', adjustable='box')
    # # plt.draw()
    plt.show()
    ans = input("Would you like to tweak some settings for better results then ? (y/n) [default to yes] : ")
    if ans == "y":
        flag = True
    elif ans == "n":
        ans = input("Would you like to create the gif ? (y/n) [defaults to yes] : ")
        if ans == "n":
            exit()
        flag = False
    else:
         flag = True

fig = plt.figure()
m = 1.05
temp_var2 = temp_var1[1].split(",")
def scale_for_min(input_for_scale):
    if input_for_scale > 0:
        return input_for_scale/m
    elif input_for_scale <= 0:
        return m*input_for_scale
def scale_for_max(input_for_scale):
    if input_for_scale >= 0:
        return input_for_scale*m
    elif input_for_scale < 0:
        return input_for_scale/m
x_min, x_max, y_min, y_max = scale_for_min(float(temp_var2[0])), scale_for_max(float(temp_var2[1])), scale_for_min(float(temp_var2[2])), scale_for_max(float(temp_var2[3]))
ax = plt.axes(xlim=(x_min, x_max), ylim=(y_min, y_max))
# print(str(x_min), str(x_max), str(y_min), str(y_max))
ax.set_aspect('equal')
line, = ax.plot([], [], lw=2)
Line, = ax.plot([], [], "r.")

ans = input("Would you like to put input points in the background ? (y/n) [defaults to yes] : ")
if ans == "n":
    x0 = []
    y0 = []
plt.axis("off")
ans = input("Would you like to put axes in the background ? (y/n) [defaults to no] : ")
if ans == "y":
    plt.axis("on")
def init():
    Line.set_data(x0, y0)
    return Line,

xdata, ydata = [], []

def animate(i):
    xdata.append(x[i])
    ydata.append(y[i])
    line.set_data(xdata, ydata)
    return line,

print("Creating the gif. Please wait...")
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=int(argv3), interval=frametime, blit=True)
anim.save("drawing.gif", writer="imagemagick")

print("The gif is saved :D")
