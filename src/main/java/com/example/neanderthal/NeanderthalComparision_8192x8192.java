package com.example.neanderthal;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

public class NeanderthalComparision_8192x8192 {


    @State(Scope.Thread)
    public static class SetupState {
        public int size = 8192;
        public INDArray m1 = Nd4j.createUninitialized(new int[]{size, size}, 'f');
        public INDArray m2 = Nd4j.createUninitialized(m1.shape(), 'f');
        public INDArray r = Nd4j.createUninitialized(m1.shape(), 'f');
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.SECONDS)
    public void gemm(SetupState state) {
        Nd4j.gemm(state.m1, state.m2, state.r, false, false, 1.0, 0.0);
    }
}
