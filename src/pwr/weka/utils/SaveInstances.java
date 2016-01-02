package pwr.weka.utils;

import java.io.File;
import java.io.IOException;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class SaveInstances {
	public void Save(Instances data, String fileName) throws IOException{
		 ArffSaver saver = new ArffSaver();
		 saver.setInstances(data);
		 saver.setFile(new File(fileName));
		 saver.writeBatch();
	}
}
