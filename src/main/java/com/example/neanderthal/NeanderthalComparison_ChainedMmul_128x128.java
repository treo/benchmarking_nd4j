package com.example.neanderthal;

import clojure.java.api.Clojure;
import clojure.lang.IFn;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.openjdk.jmh.annotations.*;

import java.util.concurrent.TimeUnit;

public class NeanderthalComparison_ChainedMmul_128x128 {

    @State(Scope.Thread)
    public static class SetupState {
        public int size = 128;
        INDArray m1;
        INDArray m2;
        INDArray m3;
        INDArray m4;
        INDArray m5;
        INDArray m6;
        IFn mm;
        IFn fmap;
        IFn fge;
        Object n_m1;
        Object n_m2;
        Object n_m3;
        Object n_m4;
        Object n_m5;
        Object n_m6;

        @Setup(Level.Iteration)
        public void doSetup(){
            IFn require = Clojure.var("clojure.core", "require");
            IFn deref = Clojure.var("clojure.core", "deref");
            require.invoke(Clojure.read("uncomplicate.neanderthal.core"));
            require.invoke(Clojure.read("uncomplicate.neanderthal.native"));
            require.invoke(Clojure.read("uncomplicate.fluokitten.core"));

            fmap = Clojure.var("uncomplicate.fluokitten.core", "fmap!");
            mm = Clojure.var("uncomplicate.neanderthal.core", "mm");
            fge = Clojure.var("uncomplicate.neanderthal.native", "fge");

            IFn random = (IFn) deref.invoke(Clojure.var("clojure.core", "eval").invoke(Clojure.read(
                    "(let [splittable-random (java.util.SplittableRandom.)]\n" +
                    "  (defn random ^double [^double _]\n" +
                    "    (.nextDouble ^java.util.SplittableRandom splittable-random)))"
            )));

            n_m1 = fmap.invoke(random, fge.invoke(size, size));
            n_m2 = fmap.invoke(random, fge.invoke(size, size));
            n_m3 = fmap.invoke(random, fge.invoke(size, size));
            n_m4 = fmap.invoke(random, fge.invoke(size, size));
            n_m5 = fmap.invoke(random, fge.invoke(size, size));
            n_m6 = fmap.invoke(random, fge.invoke(size, size));

            m1 = Nd4j.rand(size, size);
            m2 = Nd4j.rand(m1.shape());
            m3 = Nd4j.rand(m1.shape());
            m4 = Nd4j.rand(m1.shape());
            m5 = Nd4j.rand(m1.shape());
            m6 = Nd4j.rand(m1.shape());
        }
    }


    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void nd4j_mmul_chain(SetupState state) {
        state.m1.mmul(state.m2).mmul(state.m3).mmul(state.m4).mmul(state.m5).mmul(state.m6);
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void nd4j_mmuli_chain(SetupState state) {
        state.m1.mmul(state.m2).mmuli(state.m3).mmuli(state.m4).mmuli(state.m5).mmuli(state.m6);
    }

    @Benchmark @BenchmarkMode(Mode.SampleTime) @OutputTimeUnit(TimeUnit.MICROSECONDS)
    public void neanderthal_mm_chain(SetupState state) {
       state.mm.invoke(state.n_m1,state.n_m2,state.n_m3,state.n_m4,state.n_m5,state.n_m6);
    }
}
