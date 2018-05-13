package de.embl.cba.trainableDeepSegmentation;

public class ResultsTableTest
{

    public static void main( String... args )
    {
        ij.measure.ResultsTable resultsTable = new ij.measure.ResultsTable();

        if ( resultsTable.getCounter() == 0 )
        {
            resultsTable.incrementCounter();
        }

        resultsTable.addValue( "column1", "value1" );
        resultsTable.addValue( "column2", "value1" );

        resultsTable.show( "aaa" );
    }


}
