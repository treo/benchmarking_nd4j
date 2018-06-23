package com.example.bypass;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.openblas;
import org.nd4j.linalg.api.blas.Level3;
import org.nd4j.linalg.api.blas.params.GemmParams;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.aggregates.impl.AggregateGEMM;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.concurrent.TimeUnit;

import static org.bytedeco.javacpp.openblas.*;
import static org.bytedeco.javacpp.openblas.CblasConjTrans;
import static org.bytedeco.javacpp.openblas.CblasNoTrans;

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

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.NANOSECONDS)
    public void fake_sgemm(SetupState state, Blackhole bh) {
        final GemmParams params = state.params;
        if (!Nd4j.isFallbackModeEnabled()) {
            final int a1 = convertOrder('f');
            final int a2 = convertTranspose(params.getTransA());
            final int a3 = convertTranspose(params.getTransB());
            final FloatPointer b1 = (FloatPointer) params.getA().data().addressPointer();
            final FloatPointer b2 = (FloatPointer) params.getB().data().addressPointer();
            final FloatPointer b3 = (FloatPointer) params.getC().data().addressPointer();
            bh.consume(a1);
            bh.consume(a2);
            bh.consume(a3);
            bh.consume(b1);
            bh.consume(b2);
            bh.consume(b3);
        } else {
            Nd4j.getExecutioner()
                    .exec(new AggregateGEMM('f', 0, 0, 0, 0, 0, 0, null, 0, null, 0, 0, null, 0));
        }

    }


    static int convertOrder(int from) {
        switch (from) {
            case 'c':
            case 'C':
                return CblasRowMajor;
            case 'f':
            case 'F':
                return CblasColMajor;
            default:
                return CblasColMajor;
        }
    }

    /**
     * Converts a character to its proper enum
     * t -> transpose
     * n -> no transpose
     * c -> conj
     */
    static int convertTranspose(int from) {
        switch (from) {
            case 't':
            case 'T':
                return CblasTrans;
            case 'n':
            case 'N':
                return CblasNoTrans;
            case 'c':
            case 'C':
                return CblasConjTrans;
            default:
                return CblasNoTrans;
        }
    }
}
