package com.example;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Exp;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class NativeOps {

    @State(Scope.Thread)
    public static class SetupState {
        INDArray array = Nd4j.ones(1024, 1024);
        INDArray arrayRow = Nd4j.linspace(1, 1024, 1024);
        INDArray array1 = Nd4j.linspace(1, 20480, 20480);
        INDArray array2 = Nd4j.linspace(1, 20480, 20480);

        {
            float sum = (float) array.sumNumber().doubleValue();
        }
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void broadcastOp(SetupState state) throws IOException {
        state.array.addiRowVector(state.arrayRow);
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void reduceOp1(SetupState state) throws IOException {
        state.array.sum(0);
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void reduceOp2(SetupState state) throws IOException {
        state.array.sumNumber().floatValue();
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void transformOp(SetupState state) throws IOException {
        Nd4j.getExecutioner().exec(new Exp(state.array1, state.array2));
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void pairwiseOp1(SetupState state) throws IOException {
        state.array1.addiRowVector(state.array2);
    }


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void scalarOp1(SetupState state) throws IOException {
        state.array2.addi(0.5f);
    }


}
