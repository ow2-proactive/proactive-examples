// Copyright (c) 2009-2013 Intel Corporation
// All rights reserved.
//
// WARRANTY DISCLAIMER
//
// THESE MATERIALS ARE PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL INTEL OR ITS
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THESE
// MATERIALS, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Intel Corporation is the author of the Materials, and requests that all
// problem reports or change requests be submitted to it directly


#ifdef __DO_FLOAT__
#define tfloat float
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define tfloat double
#endif

#define HALF 0.5
#define ZERO 0.0
#define ONE  1.0
#define TWO  2.0

// scaling factor 2^32
#define DIVISOR 4294967296.0

#ifndef M_PI
#define M_PI 3.14159265358979
#endif

#define UINT_THIRTY 30U
#define UINT_BIG_CONST 1812433253U


// Mersenne twister algorithm constants. Please refer for details
// http://en.wikipedia.org/wiki/Mersenne_twister
// Matsumoto, M.; Nishimura, T. (1998). "Mersenne twister: a 623-dimensionally equidistributed uniform pseudo-random number generator".
// ACM Transactions on Modeling and Computer Simulation 8 (1): 3–30.

// degree of Mersenne twister recurrence
#define N 624
// middle word
#define M 397

// Mersenne twister constant 2567483615
#define MATRIX_A 0x9908B0DFU
// Mersenne twister constant 2636928640
#define MASK_B 0x9D2C5680U
// Mersenne twister constant 4022730752
#define MASK_C 0xEFC60000U
// Mersenne twister tempering shifts
#define SHIFT_U 11
#define SHIFT_S 7
#define SHIFT_T 15
#define SHIFT_L 18


kernel
    void MonteCarloEuroOptCLKernelScalarBoxMuller(
    global tfloat * restrict vcall, // option call value (OUT)
    global tfloat * restrict vput, // option put value (OUT)
    tfloat r, // risk free (IN)
    tfloat sigma, // volatility (IN)
    global const tfloat *s_price, // current stock price (IN)
    global const tfloat *k_price, // option strike price (IN)
    global const tfloat *t // time (IN)
    )
{
    // Global ID
    int tid;

    // Get global id = index of put&call options pair to be calculated
    tid = get_global_id(0);

    // Precalculate auxiliary variables
    tfloat int_to_float_normalize_factor = ONE/((tfloat)DIVISOR); // for float random number scaling

    // Auxiliary variables for share price calculation formula
    // Share price at the time of option maturity
    //  S(T) = S(0) exp(  (r-1/2  sigma^2)T + sigma * W(T) ),
    //  W is the Wiener process, that is W(T) is Gaussian variable with mean 0 and variance T
    tfloat tmp_bs1 = (r - sigma*sigma*HALF)*t[tid]; // formula reference: (r - (sigma^2)/2)*(T)
    tfloat tmp_bs2 = sigma*sqrt(t[tid]); // formula reference: sigma * (T)^(1/2)

    // Initialize options price
    vcall[tid]   = (tfloat)ZERO;
    vput[tid]    = (tfloat)ZERO;


    // State indexes
    int i, iCurrentMiddle, iCurrent;
    // Mersenne twister generated random number
    uint mt_rnd_num;
    // State of the MT generator
    int mt_state[N];
    // Temporary state for MT states swap
    int tmp_mt;

    // set seed
    mt_state[0] = tid;


    // Initialize the MT generator from a seed
    for (i = 1; i < N; i++)
    {
        mt_state[i] = (uint)i + UINT_BIG_CONST * (mt_state[i - 1] ^ (mt_state[i - 1] >> UINT_THIRTY));
    }

    // Initialize MT state
    i = 0;
    tmp_mt = mt_state[0];

    for (int iSample = 0; iSample < NSAMP; iSample = iSample + 2) // Generate two samples per iteration as it is convinient for Box-Muller
    {
        // Mersenne twister loops generating untempered and tempered values in original description merged here together with Box-Muller
        // normally distributed random numbers generation and Black&Scholes formula.

        // First MT random number generation
        // Calculate new state indexes
        iCurrent = (iCurrent == N - 1) ?  0 : i + 1;
        iCurrentMiddle = (i + M >= N) ? i + M - N : i + M;

        mt_state[i] = tmp_mt;
        tmp_mt = mt_state[iCurrent];

        // MT recurrence
        // Generate untempered numbers
        mt_rnd_num = (mt_state[i] & 0x80000000U) | (mt_state[iCurrent] & 0x7FFFFFFFU);
        mt_rnd_num = mt_state[iCurrentMiddle] ^ (mt_rnd_num >> 1) ^ ((0-(mt_rnd_num & 1))& MATRIX_A);

        mt_state[i] = mt_rnd_num;

        // Tempering pseudorandom number
        mt_rnd_num ^= (mt_rnd_num >> SHIFT_U);
        mt_rnd_num ^= (mt_rnd_num << SHIFT_S) & MASK_B;
        mt_rnd_num ^= (mt_rnd_num << SHIFT_T) & MASK_C;
        mt_rnd_num ^= (mt_rnd_num >> SHIFT_L);

        tfloat rnd_num = (tfloat)mt_rnd_num;

        i = iCurrent;

        // Second MT random number generation
        // Calculate new state indexes
        iCurrent = (iCurrent == N - 1) ?  0 : i + 1;
        iCurrentMiddle = (i + M >= N) ? i + M - N : i + M;

        mt_state[i] = tmp_mt;
        tmp_mt = mt_state[iCurrent];

        // MT recurrence
        // Generate untempered numbers
        mt_rnd_num = (mt_state[i] & 0x80000000U) | (mt_state[iCurrent] & 0x7FFFFFFFU);
        mt_rnd_num = mt_state[iCurrentMiddle] ^ (mt_rnd_num >> 1) ^ ((0-(mt_rnd_num & 1))& MATRIX_A);


        mt_state[i] = mt_rnd_num;


        // Tempering pseudorandom number
        mt_rnd_num ^= (mt_rnd_num >> SHIFT_U);
        mt_rnd_num ^= (mt_rnd_num << SHIFT_S) & MASK_B;
        mt_rnd_num ^= (mt_rnd_num << SHIFT_T) & MASK_C;
        mt_rnd_num ^= (mt_rnd_num >> SHIFT_L);

        tfloat rnd_num1 = (tfloat)mt_rnd_num;

        i = iCurrent;

        // Make uniform random variables in (0,1] range
        rnd_num = (rnd_num + ONE) * int_to_float_normalize_factor;
        rnd_num1 = (rnd_num1 + ONE) * int_to_float_normalize_factor;

        // Generate normally distributed random numbers
        // Box-Muller
        tfloat tmp_bm = sqrt(max(-TWO*log(rnd_num), 0.0)); // max added to be sure that sqrt argument non-negative
        rnd_num = tmp_bm*cos(TWO*M_PI*rnd_num1);
        rnd_num1 = tmp_bm*sin(TWO*M_PI*rnd_num1);


        // Stock price formula
        // Add first sample from pair
        tfloat tmp_bs3 = rnd_num*tmp_bs2 + tmp_bs1; // formula reference: NormalDistribution*sigma*(T)^(1/2) + (r - (sigma^2)/2)*(T)
        tmp_bs3 = s_price[tid]*exp(tmp_bs3); // formula reference: S * exp(CND*sigma*(T)^(1/2) + (r - (sigma^2)/2)*(T))


        tfloat dif_call = tmp_bs3-k_price[tid];


        vcall[tid] += max(dif_call, (tfloat)ZERO);

        // Add second sample from pair
        tmp_bs3 = rnd_num1*tmp_bs2 + tmp_bs1;
        tmp_bs3 = s_price[tid]*exp(tmp_bs3);


        dif_call = tmp_bs3-k_price[tid];

        vcall[tid] += max(dif_call, (tfloat)ZERO);

    }

    // Average
    vcall[tid] = vcall[tid] / ((tfloat)NSAMP) * exp(-r*t[tid]);


    // Calculate put option price from call option price: put = call – S0 + K * exp( -rT )
    vput[tid] = vcall[tid] - s_price[tid] + k_price[tid] * exp(-r*t[tid]);
}
