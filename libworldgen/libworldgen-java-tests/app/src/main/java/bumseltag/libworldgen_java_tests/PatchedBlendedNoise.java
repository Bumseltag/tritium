package bumseltag.libworldgen_java_tests;

import net.minecraft.util.Mth;
import net.minecraft.util.RandomSource;
import net.minecraft.world.level.levelgen.XoroshiroRandomSource;
import net.minecraft.world.level.levelgen.synth.ImprovedNoise;
import net.minecraft.world.level.levelgen.synth.PerlinNoise;

import java.util.stream.IntStream;

/**
 * A patched version of {@link net.minecraft.world.level.levelgen.synth.BlendedNoise},
 * with unnecessary static init stuff removed, since that can cause issues when calling from rust.
 */
@SuppressWarnings("deprecation")
public class PatchedBlendedNoise {
    private final PerlinNoise minLimitNoise;
    private final PerlinNoise maxLimitNoise;
    private final PerlinNoise mainNoise;
    private final double xzMultiplier;
    private final double yMultiplier;
    private final double xzFactor;
    private final double yFactor;
    private final double smearScaleMultiplier;
    private final double maxValue;
    private final double xzScale;
    private final double yScale;

    public static PatchedBlendedNoise createUnseeded(double p_230478_, double p_230479_, double p_230480_, double p_230481_, double p_230482_) {
        return new PatchedBlendedNoise(new XoroshiroRandomSource(0L), p_230478_, p_230479_, p_230480_, p_230481_, p_230482_);
    }

    private PatchedBlendedNoise(PerlinNoise p_230469_, PerlinNoise p_230470_, PerlinNoise p_230471_, double p_230472_, double p_230473_, double p_230474_, double p_230475_, double p_230476_) {
        this.minLimitNoise = p_230469_;
        this.maxLimitNoise = p_230470_;
        this.mainNoise = p_230471_;
        this.xzScale = p_230472_;
        this.yScale = p_230473_;
        this.xzFactor = p_230474_;
        this.yFactor = p_230475_;
        this.smearScaleMultiplier = p_230476_;
        this.xzMultiplier = 684.412D * this.xzScale;
        this.yMultiplier = 684.412D * this.yScale;
        this.maxValue = p_230469_.maxBrokenValue(this.yMultiplier);
    }

    public PatchedBlendedNoise(RandomSource p_230462_, double p_230463_, double p_230464_, double p_230465_, double p_230466_, double p_230467_) {
        this(PerlinNoise.createLegacyForBlendedNoise(p_230462_, IntStream.rangeClosed(-15, 0)), PerlinNoise.createLegacyForBlendedNoise(p_230462_, IntStream.rangeClosed(-15, 0)), PerlinNoise.createLegacyForBlendedNoise(p_230462_, IntStream.rangeClosed(-7, 0)), p_230463_, p_230464_, p_230465_, p_230466_, p_230467_);
    }

    public PatchedBlendedNoise withNewRandom(RandomSource p_230484_) {
        return new PatchedBlendedNoise(p_230484_, this.xzScale, this.yScale, this.xzFactor, this.yFactor, this.smearScaleMultiplier);
    }

    public double compute(double x, double y, double z) {
        double d0 = x * this.xzMultiplier;
        double d1 = y * this.yMultiplier;
        double d2 = z * this.xzMultiplier;
        double d3 = d0 / this.xzFactor;
        double d4 = d1 / this.yFactor;
        double d5 = d2 / this.xzFactor;
        double d6 = this.yMultiplier * this.smearScaleMultiplier;
        double d7 = d6 / this.yFactor;
        double d8 = 0.0D;
        double d9 = 0.0D;
        double d10 = 0.0D;
        boolean flag = true;
        double d11 = 1.0D;

        for(int i = 0; i < 8; ++i) {
            ImprovedNoise improvednoise = this.mainNoise.getOctaveNoise(i);
            if (improvednoise != null) {
                d10 += improvednoise.noise(PerlinNoise.wrap(d3 * d11), PerlinNoise.wrap(d4 * d11), PerlinNoise.wrap(d5 * d11), d7 * d11, d4 * d11) / d11;
            }

            d11 /= 2.0D;
        }

        double d16 = (d10 / 10.0D + 1.0D) / 2.0D;
        boolean flag1 = d16 >= 1.0D;
        boolean flag2 = d16 <= 0.0D;
        d11 = 1.0D;

        for(int j = 0; j < 16; ++j) {
            double d12 = PerlinNoise.wrap(d0 * d11);
            double d13 = PerlinNoise.wrap(d1 * d11);
            double d14 = PerlinNoise.wrap(d2 * d11);
            double d15 = d6 * d11;
            if (!flag1) {
                ImprovedNoise improvednoise1 = this.minLimitNoise.getOctaveNoise(j);
                if (improvednoise1 != null) {
                    d8 += improvednoise1.noise(d12, d13, d14, d15, d1 * d11) / d11;
                }
            }

            if (!flag2) {
                ImprovedNoise improvednoise2 = this.maxLimitNoise.getOctaveNoise(j);
                if (improvednoise2 != null) {
                    d9 += improvednoise2.noise(d12, d13, d14, d15, d1 * d11) / d11;
                }
            }

            d11 /= 2.0D;
        }

        return Mth.clampedLerp(d8 / 512.0D, d9 / 512.0D, d16) / 128.0D;
    }

    public double minValue() {
        return -this.maxValue();
    }

    public double maxValue() {
        return this.maxValue;
    }
}
