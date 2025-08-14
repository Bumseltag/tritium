package bumseltag.libworldgen_java_tests;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import net.minecraft.util.RandomSource;
import net.minecraft.world.level.levelgen.WorldgenRandom;
import net.minecraft.world.level.levelgen.synth.NormalNoise;
import org.jetbrains.annotations.NotNull;

import java.util.function.Function;
import java.util.function.Supplier;

public class App {
    public static void main(String @NotNull [] args) {
        JsonObject obj = new JsonObject();
        obj.add("legacy_rng_0", testRng(0, WorldgenRandom.Algorithm.LEGACY));
        obj.add("legacy_rng_1", testRng(1, WorldgenRandom.Algorithm.LEGACY));
        obj.add("xoroshiro_rng_0", testRng(0, WorldgenRandom.Algorithm.XOROSHIRO));
        obj.add("xoroshiro_rng_1", testRng(1, WorldgenRandom.Algorithm.XOROSHIRO));
    }

    public static JsonElement testRng(long seed, WorldgenRandom.Algorithm rngAlgorithm) {
        RandomSource rng = rngAlgorithm.newInstance(seed);
        JsonObject res = new JsonObject();
        res.add("next_int", repeatNum(5, rng::nextInt));
        res.add("next_int_up_to_4", repeatNum(5, () -> rng.nextInt(4)));
        res.add("next_int_between_3_and_5", repeatNum(5, () -> rng.nextInt(3, 5)));
        res.add("next_long", repeatNum(5, rng::nextLong));
        res.add("next_float", repeatNum(5, rng::nextFloat));
        res.add("next_double", repeatNum(5, rng::nextDouble));
        res.add("next_bool", repeatBool(5, rng::nextBoolean));
        res.add("next_gaussian", repeatNum(5, rng::nextGaussian));
        res.add("next_triangle_2_4", repeatNum(5, () -> rng.triangle(2, 4)));
        return res;
    }

    public static JsonElement testNormalNoise(
            long seed,
            WorldgenRandom.Algorithm rngAlgorithm,
            NormalNoise.NoiseParameters params
    ) {
        RandomSource rng = rngAlgorithm.newInstance(seed);
        NormalNoise noise = NormalNoise.create(rng, params);
        JsonArray res = new JsonArray();
        for (int x = -5; x <= 5; x++) {
            JsonArray xRes = new JsonArray();
            for (int y = -5; y <= 5; y++) {
                JsonArray yRes = new JsonArray();
                for (int z = -5; z <= 5; z++) {
                    yRes.add(noise.getValue(x, y, z));
                }
                xRes.add(yRes);
            }
            res.add(xRes);
        }
        return res;
    }

    public static JsonArray forRangeNum(int from, int to, Function<Integer, Number> fn) {
        JsonArray res = new JsonArray();
        for (int i = from; i <= to; i++) {
            res.add(fn.apply(i));
        }
        return res;
    }

    public static JsonArray repeatNum(int n, Supplier<Number> fn) {
        return forRangeNum(0, n - 1, (i) -> fn.get());
    }

    public static JsonArray forRangeBool(int from, int to, Function<Integer, Boolean> fn) {
        JsonArray res = new JsonArray();
        for (int i = from; i <= to; i++) {
            res.add(fn.apply(i));
        }
        return res;
    }

    public static JsonArray repeatBool(int n, Supplier<Boolean> fn) {
        return forRangeBool(0, n - 1, (i) -> fn.get());
    }
}
