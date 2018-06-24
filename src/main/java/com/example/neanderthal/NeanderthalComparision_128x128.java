package com.example.neanderthal;

import clojure.java.api.Clojure;
import clojure.lang.IFn;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

public class NeanderthalComparision_128x128 {


    @State(Scope.Thread)
    public static class SetupState {
        public int size;
        public INDArray m1;
        public INDArray m2;
        public INDArray r;
        IFn mm;
        IFn fge;
        Object n_m1;
        Object n_m2;
        Object n_r;

        @Setup(Level.Trial)
        public void doSetup(){
            IFn require = Clojure.var("clojure.core", "require");
            require.invoke(Clojure.read("uncomplicate.neanderthal.core"));
            require.invoke(Clojure.read("uncomplicate.neanderthal.native"));

            size = 128;
            long sizel = 128;

            mm = Clojure.var("uncomplicate.neanderthal.core", "mm!");
            fge = Clojure.var("uncomplicate.neanderthal.native", "fge");

            n_m1 = fge.invoke(sizel, sizel);
            n_m2 = fge.invoke(sizel, sizel);
            n_r = fge.invoke(sizel, sizel);

            m1 = Nd4j.ones(size, size);
            m2 = Nd4j.ones(m1.shape());
            r = Nd4j.createUninitialized(m1.shape(), 'f');
        }
    }


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void nd4j_gemm(SetupState state) {
        Nd4j.gemm(state.m1, state.m2, state.r, false, false, 1.0, 0.0);
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void neanderthal_mm(SetupState state) {
       state.mm.invoke(1.0f, state.n_m1, state.n_m2, 0.0f, state.n_r);
    }
}
