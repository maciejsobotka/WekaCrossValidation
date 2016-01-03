package pwr.weka.main;

import pwr.weka.crossvalidation.CrossValidation;
import pwr.weka.utils.LoadInstances;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

/**
 * Place where the program starts.
 * Two available arguments
 * 1 - number of folds to use in cross-validation
 * 2 - number of cross-validation repeats
 * 
 * @author Sobot
 */
public class ProgramMain {
	public static void main(String[] args){
		int numFolds = 10;
		int numReps = 1;
		if(args.length == 1) {numFolds = Integer.parseInt(args[0]); numReps = 1;}
		if(args.length == 2) {
			numFolds = Integer.parseInt(args[0]);
			numReps = Integer.parseInt(args[1]);
		}
		
		LoadInstances loader = new LoadInstances();
		Instances data;
		try {
			data = loader.Load("195420L4_1.arff");
			CrossValidation cv = new CrossValidation();
			Classifier classifier = (Classifier)new NaiveBayes();
			cv.doCrossValidation(data, classifier , numFolds, numReps);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
