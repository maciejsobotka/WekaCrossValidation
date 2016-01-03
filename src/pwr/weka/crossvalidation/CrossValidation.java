package pwr.weka.crossvalidation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class CrossValidation {
	private Instances[] folds;
	
	public void doCrossValidation(Instances data, Classifier classifier, int numFolds, int numReps){

		generateFolds(data, numFolds);
		Instances trainingSet = new Instances(data, data.numInstances() - data.numInstances()/numFolds);
		Instances testSet = new Instances(data, data.numInstances()/numFolds+1);
		int n;
		double[][] cmMatrix = null;
		try {
			for(int k = 0; k < numReps; ++k){					// number of repetitions
				n = numFolds;
				for(int j = 0; j < numFolds; ++j){				// number of folds
					n--;
					trainingSet.clear();
					for(int i = 0; i < folds.length; ++i)		// trainingSet creation
						if (i != n)	trainingSet.addAll(folds[i]);
					testSet = folds[n];							// testSet creation
					classifier.buildClassifier(trainingSet);	// part of cross-validation
					Evaluation eval = new Evaluation(trainingSet);
					eval.evaluateModel(classifier, testSet);
					double[][] cmPartMatrix = eval.confusionMatrix();
					if (j == 0) cmMatrix = cmPartMatrix;
					else cmMatrix = summConfMatrixs(cmMatrix, cmPartMatrix);
				}
				printConfMatrix(cmMatrix);
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void generateFolds(Instances data, int numFolds){

		folds = new Instances[numFolds];
		for(int i=0; i<folds.length; ++i)
			folds[i] = new Instances(data, data.numInstances()/numFolds+1);
		List<Integer> rand = new ArrayList<>();
		for(int i=0; i<data.numInstances(); ++i)
			rand.add(i);
		Collections.shuffle(rand);
		
		int n=0;
		for(int i=0; i<data.numInstances(); ++i){
			folds[n].add(data.get(rand.get(i)));
			if(n==9) n=0;
			else n++;
		}
	}
	private double[][] summConfMatrixs(double[][] cmMatrix1, double[][] cmMatrix2){
		
		double[][] sumConfMatrix = new double[cmMatrix1.length][cmMatrix1.length];
		for(int i = 0; i < cmMatrix1.length; ++i)
			for(int j = 0; j < cmMatrix1.length; ++j)
				sumConfMatrix[i][j] = cmMatrix1[i][j] + cmMatrix2[i][j];
		return sumConfMatrix;
	}
	
	private void printConfMatrix(double[][] cmMatrix){
		
		for(int i = 0; i < cmMatrix.length; ++i){
			for(int j = 0; j < cmMatrix.length; ++j){
				System.out.print(cmMatrix[i][j]);
				System.out.print(' ');					
			}
			System.out.println();
		}
	}
}
