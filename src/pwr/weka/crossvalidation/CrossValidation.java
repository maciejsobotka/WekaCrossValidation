package pwr.weka.crossvalidation;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import weka.classifiers.Classifier;
import weka.core.Instances;

public class CrossValidation {
	private Instances[] folds;
	
	public void doCrossValidation(Instances data, Classifier classifier, int numFolds, int numReps){
		generateFolds(data, numFolds);
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
}
