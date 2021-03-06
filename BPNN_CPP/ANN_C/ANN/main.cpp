// ANN.cpp: 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "cstdio"
#include "iostream"
#include "cmath"
#include "vector"
#include "algorithm"
#include "ANN.h"
#include "time.h"
#include "sstream"
#include "fstream"
using namespace std;

void print(double x)
{
	cout << x << endl;
}
//输出各层参数
void disp(ANN& a) {
	struct Layer *ll = a.getLayers();
	int numL = a.getNumLayer(), numP, numI;
	for (int i = 1; i < numL; i++) {
		cout << "Layer " << i << ":" << endl;
		numP = ll[i].perceptronNum;
		for (int j = 0; j < numP; j++) {
			cout << "\tPerception " << j << ":\t";
			numI = ll[i].perceptrons[j].inputWeightNum;
			for (int k = 0; k < numI; k++) {
				cout << ll[i].perceptrons[j].inputWeights[k] << "   ";
			}
			cout << endl;
		}
		cout << "\n\n";
	}
	system("pause");
}


int main()
{
	srand((unsigned)time(NULL));
	int layerArr[] = {400,300,100,10};
	vector<int> layerSS(layerArr, layerArr + sizeof(layerArr)/sizeof(int));
	//初始化神经网络
	ANN a(layerSS);

	//设置参数
	a.setParameter(0.2, 100, 0.01);
	double accuracy = 0;
	string fileName = "4_300_100";
	//若已存在网络，加载文件参数
	a.loadANN("./ann_"+ fileName +".parameter");
	int epoch = 100;

	//训练epoch代
	clock_t startTime, endTime,s,e;
	for(int i=0;i<epoch;i++){
		s = clock();
		//一代 60 个数据集
		for (int j = 1; j < 61; j++) {
			//加载1个数据集
			a.addData(j);
			//训练
			a.train();
			//学习率随迭代次数增长而下降
			a.setLearningRate(0.1 / (1 + i * 0.001));
			cout <<"set:"<<j<<"\t"<< a.getSquareError() << endl;
		}
		e = clock();
		//时间
		cout << "##################one epoch time:  " << (double)(e - s) / CLOCKS_PER_SEC <<"S "<< endl;

		stringstream ss;
		ss << i;
		string str = ss.str();
		//存储网络参数
		a.saveANN("./ann_"+fileName+".parameter");
		ofstream out("./test_"+fileName+".txt",ios::app);
		if (out.is_open())
		{
			accuracy = a.test();
			out << "Epoch: " << i + 1 << " " << accuracy << "%" << endl;
			out.close();
		}
		cout << "Epoch: "<<i+1<<" " << accuracy<<"%" << endl;
	}
    return 0;
}


