# include "stdio.h"
# include "conio.h"
# include "windows.h"


//原始的图表，五行六列，其中 0 代表着空白的地方； 1 代表着墙；2 代表着人；3 代表着箱子；4 代表着箱子的终点位置。
int map[9][11] = {
	{0,1,1,1,1,1,1,1,1,1,0},
	{0,1,0,0,0,1,0,0,0,1,0},
	{0,1,0,3,3,3,3,3,0,1,0},
	{0,1,0,3,0,3,0,3,0,1,1},
	{0,1,0,0,0,2,0,0,3,0,1},
	{1,1,0,1,1,1,1,0,3,0,1},
	{1,0,4,4,4,4,4,1,0,0,1},
	{1,0,4,4,4,4,4,0,0,1,1},
	{1,1,1,1,1,1,1,1,1,1,0}
	}; 

int draw();
int push();
int referee();

int main()
{
    while (1)
    {
        system("cls");
        draw();
        push();
    }
    printf("output:\n");
    return 0;
}
 
int draw()
{
    int i, j;
    referee();
    for(i=0; i<9; i++)
    {
        for(j=0; j<11; j++)
        {
            switch (map[i][j])
            {
                case 0:
                    printf(" ");  //空白
                    break;
                case 1:
                    printf("■"); //墙
                    break;
                case 2:
                    printf("♀");  //人
                    break;
                case 3:
                    printf("☆");  //箱子
                    break;
                case 4:
                    printf("◎");  //终点
                    break;
                case 6:
                    printf("♂");  //人到终点
                    break;
                case 7:
                    printf("★");  //箱子到终点
                    break;
            }
        }
        printf("\n");  //一行结束
    }
}   

int push()
{
    //人的位置
    int row, col;
    for(int i=0; i<9; i++)
    {
        for(int j=0; j<9; j++)
        {
            if(map[i][j] == 2 || map[i][j] == 6)
            {
                row = i;
                col = j;
            }
        }
    }
    //移动
    int move = getch();  //getchar()输入一个字符后需要回车来进行下一个字符的输入，getch()则不需要回车就能连续输入多个字符。
    switch (move)
    {
        case 'W':
        case 72:
            if(map[row-1][col] == 0 || map[row-1][col] == 4)  //下一步为空地或者终点
            {
                map[row][col] -= 2;  
                map[row-1][col] += 2;  //人往上走一步
            }
            else if(map[row-1][col] == 3 || map[row-1][col] == 7)  //下一步为箱子或箱子+终点
            {
                if(map[row-2][col] == 0 || map[row-2][col] == 4)  //箱子下一步为空地或终点
                {
                    map[row][col] -= 2;  
                    map[row-1][col] -= 1;
                    map[row-2][col] += 3;  //箱子往上一步人往上一格
                }
            }
            break;

        case 'S':
        case 80:
            if(map[row+1][col] == 0 || map[row+1][col] == 4)  //下一步为空地或者终点
            {
                map[row][col] -= 2;  
                map[row+1][col] += 2;  //人往下走一步
            }
            else if(map[row+1][col] == 3 || map[row+1][col] == 7)  //下一步为箱子或箱子+终点
            {
                if(map[row+2][col] == 0 || map[row+2][col] == 4)  //箱子下一步为空地或终点
                {
                    map[row][col] -= 2;  
                    map[row+1][col] -= 1;
                    map[row+2][col] += 3;  //箱子往下一步人往下一格
                }
            }
            break;

        case 'D':
        case 75:
            if(map[row][col+1] == 0 || map[row][col+1] == 4)  //下一步为空地或者终点
            {
                map[row][col] -= 2;  
                map[row][col+1] += 2;  //人往右走一步
            }
            else if(map[row][col+1] == 3 || map[row][col+1] == 7)  //下一步为箱子或箱子+终点
            {
                if(map[row][col+2] == 0 || map[row][col+2] == 4)  //箱子下一步为空地或终点
                {
                    map[row][col] -= 2;  
                    map[row][col+1] -= 1;
                    map[row][col+2] += 3;  //箱子往右一步人往右一格
                }
            }
            break;

        case 'A':
        case 77:
            if(map[row][col-1] == 0 || map[row][col-1] == 4)  //下一步为空地或者终点
            {
                map[row][col] -= 2;  
                map[row][col-1] += 2;  //人往左走一步
            }
            else if(map[row][col-1] == 3 || map[row][col-1] == 7)  //下一步为箱子或箱子+终点
            {
                if(map[row][col-2] == 0 || map[row][col-2] == 4)  //箱子下一步为空地或终点
                {
                    map[row][col] -= 2;  
                    map[row][col-1] -= 1;
                    map[row][col-2] += 3;  //箱子往左一步人往左一格
                }
            }
            break;    
    }
    
}

int referee()
{
    int k = 0;
    for(int i=0; i<9; i++)
    {
        for(int j=0; j<9; j++)
        {
            if(map[i][j] == 3)
            {
                k++;
            }
        }
    }
    if(k == 0)
    {
        printf("YOU WIN !!!\n");
    }
}
