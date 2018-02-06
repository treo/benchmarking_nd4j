package com.example.neanderthal;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

public class NeanderthalComparision_1024x1024 {


    @State(Scope.Thread)
    public static class SetupState {
        public int size = 1024;
        public INDArray m1 = Nd4j.ones(size, size);
        public INDArray m2 = Nd4j.ones(m1.shape());
        public INDArray r = Nd4j.createUninitialized(m1.shape(), 'f');
    }


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void mmuli(SetupState state) {
        state.m1.mmuli(state.m2, state.r);
    }


}
