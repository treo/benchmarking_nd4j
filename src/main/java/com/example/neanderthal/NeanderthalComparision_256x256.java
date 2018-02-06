package com.example.neanderthal;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

public class NeanderthalComparision_256x256 {


    @State(Scope.Thread)
    public static class SetupState {
        public int size = 256;
        public INDArray m1 = Nd4j.ones(size, size);
        public INDArray m2 = Nd4j.ones(m1.shape());
        public INDArray r = Nd4j.createUninitialized(m1.shape(), 'c');
    }


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void mmuli(SetupState state) {
        state.m1.mmuli(state.m2, state.r);
    }


}
