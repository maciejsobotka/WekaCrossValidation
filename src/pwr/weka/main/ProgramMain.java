package pwr.weka.main;

import java.io.PrintWriter;

import pwr.weka.crossvalidation.CrossValidation;
import pwr.weka.utils.LoadInstances;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.ZeroR;
import weka.classifiers.trees.J48;
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
		int numReps = 5;
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
			CrossValidation cv2 = new CrossValidation();
			CrossValidation cv3 = new CrossValidation();
			CrossValidation cv4 = new CrossValidation();
			CrossValidation cv5 = new CrossValidation();
			CrossValidation cv6 = new CrossValidation();
			
			Classifier classifier = (Classifier)new NaiveBayes();
			Classifier classifier2 = (Classifier)new ZeroR();
			JRip classifier3 = new JRip();
			JRip classifier4 = new JRip();
			classifier4.setMinNo(5.0);
			J48 classifier5 = new J48();
			J48 classifier6 = new J48();
			classifier6.setUnpruned(true);
			
			cv.doCrossValidation(data, classifier , numFolds, numReps);
			System.out.println(cv.getSummary());
			cv2.doCrossValidation(data, classifier2 , numFolds, numReps);
			System.out.println(cv2.getSummary());
			cv3.doCrossValidation(data, classifier3 , numFolds, numReps);
			System.out.println(cv3.getSummary());
			cv4.doCrossValidation(data, classifier4 , numFolds, numReps);
			System.out.println(cv4.getSummary());
			cv5.doCrossValidation(data, classifier5 , numFolds, numReps);
			System.out.println(cv5.getSummary());
			cv6.doCrossValidation(data, classifier6 , numFolds, numReps);
			System.out.println(cv6.getSummary());
			
			PrintWriter out = new PrintWriter("NaiveBayes.txt");
			out.println(cv.getSummary());
			out.close();
			out = new PrintWriter("ZeroR.txt");
			out.println(cv2.getSummary());
			out.close();
			out = new PrintWriter("JRip.txt");
			out.println(cv3.getSummary());
			out.close();
			out = new PrintWriter("JRip2.txt");
			out.println(cv4.getSummary());
			out.close();
			out = new PrintWriter("J48.txt");
			out.println(cv5.getSummary());
			out.close();
			out = new PrintWriter("J482.txt");
			out.println(cv6.getSummary());
			out.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
