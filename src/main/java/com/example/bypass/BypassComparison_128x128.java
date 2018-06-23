package com.example.bypass;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.openblas;
import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.blas.params.GemmParams;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.concurrent.TimeUnit;

public class BypassComparison_128x128 {


    @State(Scope.Thread)
    public static class SetupState {
        public int size = 128;
        public INDArray m1 = Nd4j.ones(size, size);
        public INDArray m2 = Nd4j.ones(m1.shape());
        public INDArray r = Nd4j.createUninitialized(m1.shape(), 'f');


        public GemmParams params = new GemmParams(m1, m2, r);
        FloatPointer a = (FloatPointer) params.getA().data().addressPointer();
        FloatPointer b = (FloatPointer) params.getB().data().addressPointer();
        FloatPointer c = (FloatPointer) params.getC().data().addressPointer();

        int M = params.getM();
        int N = params.getN();
        int K = params.getK();
        int lda = params.getLda();
        int ldb = params.getLdb();
        int ldc = params.getLdc();


        public Level3 wrapper = Nd4j.getBlasWrapper().level3();
        public Method sgemm;

        @Setup(Level.Iteration)
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


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void cblas_gemm(SetupState state, Blackhole bh) {
        openblas.cblas_sgemm(102,111, 111, state.M, state.N, state.K, 1.0f, state.a, state.lda, state.b, state.ldb, 0.0f, state.c, state.ldc);
    }


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void sgemm(SetupState state, Blackhole bh) {
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
