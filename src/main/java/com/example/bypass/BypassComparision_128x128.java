package com.example.bypass;

import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.blas.params.GemmParams;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.concurrent.TimeUnit;

public class BypassComparision_128x128 {


    @State(Scope.Thread)
    public static class SetupState {
        public int size = 128;
        public INDArray m1 = Nd4j.ones(size, size);
        public INDArray m2 = Nd4j.ones(m1.shape());
        public INDArray r = Nd4j.createUninitialized(m1.shape(), 'f');

        public Level3 wrapper = Nd4j.getBlasWrapper().level3();
        public Method sgemm;
        public GemmParams params = new GemmParams(m1, m2, r);

        @Setup(Level.Trial)
        public void doSetup(){
            try {
                sgemm = wrapper.getClass().getDeclaredMethod("sgemm", char.class, char.class, char.class, int.class, int.class, int.class, float.class, INDArray.class,
                        int.class, INDArray.class, int.class, float.class, INDArray.class, int.class);
                sgemm.setAccessible(true);
            } catch (NoSuchMethodException e) {
                e.printStackTrace();
            }
        }
    }


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.MILLISECONDS)
    public void gemm(SetupState state, Blackhole bh) {
        final GemmParams params = state.params;
        try {
            state.sgemm.invoke(state.wrapper, params.getA().ordering(), params.getTransA(), params.getTransB(), params.getM(), params.getN(), params.getK(), (float)1.0, params.getA(), params.getLda(), params.getB(), params.getLdb(), (float)0.0, params.getC(), params.getLdc());
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        }
    }



}
