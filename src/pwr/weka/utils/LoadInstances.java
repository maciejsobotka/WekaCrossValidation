package pwr.weka.utils;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class LoadInstances {
	public Instances Load(String fileName) throws Exception{
		 DataSource source = new DataSource(fileName);
		 Instances data = source.getDataSet();
		 // setting class attribute if the data format does not provide this information
		 // For example, the XRFF format saves the class attribute information as well
		 if (data.classIndex() == -1)
		   data.setClassIndex(data.numAttributes() - 1);
		 
		 return data;
	}
}
