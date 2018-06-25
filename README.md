# benchmarking_nd4j
Simple Microbenchmarks for ND4J utilizing the JMH

## Building

   mvn clean package

## Running

  java -jar target/benchmarks.jar


For a quick run, e.g. just the same Matrix Size progression as used in the Neanderthal benchmarks, run like this:

   java -jar target/benchmarks.jar -f2 -i10 -wi 2 Neanderthal
