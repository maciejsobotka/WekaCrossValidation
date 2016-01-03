package pwr.weka.crossvalidation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public class CrossValidation {
	private Instances[] folds;
	private	double acc, tPRate, fPRate, tNRate, gMean, aUC;
	private double[][] cmMatrix;
	private String summary;
	
	public void doCrossValidation(Instances data, Classifier classifier, int numFolds, int numReps) throws Exception{

		Instances trainingSet = new Instances(data, data.numInstances() - data.numInstances()/numFolds);
		Instances testSet = new Instances(data, data.numInstances()/numFolds+1);
		int n;
		double[][] cmMatrix = null;
		summary = "";
		for(int k = 0; k < numReps; ++k){					// number of repetitions
			generateFolds(data, numFolds);
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
			if (k == 0) this.cmMatrix = cmMatrix;
			else this.cmMatrix = summConfMatrixs(this.cmMatrix, cmMatrix);
		}
		this.cmMatrix = meanConfMatrix(this.cmMatrix, numReps);
		summary = printConfMatrix(this.cmMatrix);
		acc = (cmMatrix[0][0] + cmMatrix[1][1]) /
			(cmMatrix[0][0] + cmMatrix[0][1] + cmMatrix[1][0] + cmMatrix[1][1]);
		tPRate = cmMatrix[1][1] / (cmMatrix[1][0] + cmMatrix[1][1]);
		fPRate = cmMatrix[0][1] / (cmMatrix[0][1] + cmMatrix[1][1]);
		tNRate = cmMatrix[0][0] / (cmMatrix[0][0] + cmMatrix[0][1]);
		gMean = Math.sqrt(tPRate * tNRate);
		aUC = (1 + tPRate - fPRate) / 2;
		summary += statsToString();
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
	
	private double[][] meanConfMatrix(double[][] cmMatrix, int numReps){
		
		double[][] sumConfMatrix = new double[cmMatrix.length][cmMatrix.length];
		for(int i = 0; i < cmMatrix.length; ++i)
			for(int j = 0; j < cmMatrix.length; ++j)
				sumConfMatrix[i][j] = cmMatrix[i][j] / numReps;
		return sumConfMatrix;
	}
	
	private String statsToString(){
		
		String stats = "=== Classification Stats ===\n\n";
		stats += "Accuracy: " + acc + "\n";
		stats += "TPrate: " + tPRate + "\n";
		stats += "TNrate: " + tNRate + "\n";
		stats += "GMean: " + gMean + "\n";
		stats += "AUC: " + aUC + "\n";
		return stats;
	}
	
	private String printConfMatrix(double[][] cmMatrix){
		
		String matrix = "=== Confusion Matrix ===\n\n";
		for(int i = 0; i < cmMatrix.length; ++i){
			for(int j = 0; j < cmMatrix.length; ++j){
				matrix += cmMatrix[i][j] + " ";				
			}
			matrix += "\n";
		}
		matrix += "\na = neg, b = pos\n\n";
		return matrix;
	}

	public String getSummary() {
		return summary;
	}
}
