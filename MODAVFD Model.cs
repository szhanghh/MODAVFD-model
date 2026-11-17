using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;
using System.Collections.Generic;

namespace LGMD
{
    /// <summary>
    internal class LGMDs : IComparable, IComparer<LGMDs>
    {
        #region LGMD FIELD

        /* PARAMS */
        protected int width; //width of input frame
        protected int height; //height of input frame
        protected int Ncell; //number of cells(pixels)
        protected int Np; //radius in convolving matrix normally set to 1
        protected int Nsp; //number of successive spikes
        protected int Nts; //successive spikes in time steps indicating looming detection
        protected int tau_hp; //time constant in high-pass filter in milliseconds
        protected int[] tau_lp; //time constant in low-pass filter in milliseconds
        protected int Ts; //local threshold filtering S cells
        protected int Tffi; //FFI threshold
        protected int Ksp; //a scale-efficient in spiking mechanisms
        protected float clip_point; //clip point in ON and OFF rectifiers
        protected float W_on; //weight of ON cell contribution
        protected float W_off; //weight of OFF cell contribution
        protected float W_onoff; //weight of ON and OFF cells co-contribution
        protected float Tsp; //firing threshold to invoke spikes
        protected float sp_step; //firing threshold step
        protected float Tsf; //Spike-frequency adaptation threshold
        protected float W_i_on; //inhibitory bias in ON channels
        protected float W_i_off; //inhibitory bias in OFF channels
        //private float coe_ffi; //a constant to multiply ffi
        protected float dc; //a proportion in DC Component
        protected float coe_sig; //a coefficient in Sigmoid function
        protected float[,] W_l;   //local convolving matrix in ON/OFF pathways for lateral connections
        protected float[,] W_g; //local convolving matrix in grouping layer
        protected float time_interval; //time interval between frames in milliseconds
        protected float hp_delay; //time delay in HPF
        protected float[] lp_delay; //time delay in LPF

        /* MEMBERS */
        protected int[, ,] photoreceptors; //Photoreceptors after high-pass filtering
        protected float[, ,] ons; //ON cells
        protected float[, ,] offs; //OFF cells
        protected float[, ,] ONs_Compressed;  // ON channels compressed signal
        protected float[, ,] OFFs_Compressed;  // OFF channels compressed signal
        protected float[,] Kernel_Constrast; // Convolution kernel in contrast normalization
        protected float[, ,] ONs_Contrast; // Contrast vision in ON channels
        protected float[, ,] OFFs_Contrast;  // Contrast vision in OFF channels
        //private float[,] on_lp; //Delayed information in ON channel
        //private float[,] off_lp; //Delayed information in OFF channel
        protected float[,] on_inh; //inhibitions in ON channel
        protected float[,] off_inh; //excitations in OFF channel
        protected float S_on; //local ON summation cell
        protected float S_off; //local OFF summation cell
        protected float[,] scells_on; //summation cells
        protected float[,] scells_off; //summation cells
        protected float[,] gcells_on; //grouping cells
        protected float[,] gcells_off; //grouping cells
        protected float[] ffi; //feed forward inhibition in time series
        //private float ffe; //feed forward excitation
        protected float[] mp; //membrane potential
        protected float[] smp; //sigmoid membrane potential in time series
        protected float[] sfa; //SMP after spiking frequency adaptation
        protected float[] diff_smp; //differences between successive SMPs output to calculate derivative
        protected byte spike; //spiking 1,2 or not spiking 0
        protected byte collision; //collision detection true (1) or false (0)
        protected float Delta_C;  // A small real number to compute the scale in G-layer
        protected int Cw;  // A constant to compute the scale in G-layer
        protected double Cde;  // Local excitation decay coefficient
        protected int Tde;  // Local excitation decay thresholding

        //BP filters fields
        protected int sd; //sampling distance
        protected int inh_gauss_width;    //inhibitory gaussian kernel width
        protected int exc_gauss_width;    //excitatory gaussian kernel width
        protected float esd;  //excitatory standard deviation
        protected float isd;  //inhibitory standard deviation
        protected float[,] inh_gauss_kernel;  //inhibitory gaussian kernel
        protected float[,] exc_gauss_kernel;  //excitatory gaussian kernel
        protected float[,] ONs_delay;    // ON channels delayed signal - Dimension width * height * 2
        protected float[,] OFFs_delay;   // OFF channels delayed signal - Dimension width * height * 2
        protected float[, , ,] T4s;      // T4 neurons in Medulla - Dimension width * height * 2 * 4 (cardinal directions), 0->rightward, 1->leftward, 2->downward, 3->upward
        protected float[, , ,] T5s;      // T5 neurons in Lobula - Dimension width * height * 2 * 4 (cardinal directions), 0->rightward, 1->leftward, 2->downward, 3->upward
        protected float[] LPTCs;         // LPTCs in Lobula Plate - Dimension 4 * 1 vector, 0->rightward, 1->leftward, 2->downward, 3->upward
        protected float[] LPis;          // Lobula Plate intrinsic interneurons inhibiting stratified LPTCs - Dimension 4 * 1 vector, 0->leftward, 1->rightward, 2->upward, 3->downward
        protected float SysHS;           // System of horiztonal sensitive
        protected float SysVS;           // System of vertical sensitive

        #endregion

        #region GA FIELD

        /// <summary>
        /// dictionary of searching parameters in GA
        /// </summary>
        private Dictionary<string, float> paramsDict;
        /// <summary>
        /// fitness value
        /// </summary>
        private float fitness;
        /// <summary>
        /// old (true) or new (false) generation of object 
        /// </summary>
        private bool isOldGeneration;
        /// <summary>
        /// timing of activation of neuron
        /// </summary>
        private int activationTiming;
        /// <summary>
        /// belongs to best agents (true) or not (false)
        /// </summary>
        private bool isBestAgent;
        /// <summary>
        /// parent selected (true) or not (false)
        /// </summary>
        private bool isParent;
        /// <summary>
        /// failure detection of colliding scenarios
        /// </summary>
        private int Fcol;
        /// <summary>
        /// wrong collision-like response to non-collision scenarios
        /// </summary>
        private int Fnon;
        /// <summary>
        /// max value of local threshold in S layer
        /// </summary>
        public readonly int max_Ts;
        /// <summary>
        /// min value of local threshold in S layer
        /// </summary>
        public readonly int min_Ts;
        /// <summary>
        /// max value of FFI threshold
        /// </summary>
        public readonly int max_Tffi;
        /// <summary>
        /// min value of FFI threshold
        /// </summary>
        public readonly int min_Tffi;
        /// <summary>
        /// max value of spiking threshold
        /// </summary>
        public readonly float max_Tsp;
        /// <summary>
        /// min value of spiking threshold
        /// </summary>
        public readonly float min_Tsp;
        /// <summary>
        /// max value of coefficient in sigmoid transformation
        /// </summary>
        public readonly float max_Csig;
        /// <summary>
        /// min value of coefficient in sigmoid transformation
        /// </summary>
        public readonly float min_Csig;
        /// <summary>
        /// max value of local bias in ON channels
        /// </summary>
        public readonly float max_W_i_on;
        /// <summary>
        /// min value of local bias in ON channels
        /// </summary>
        public readonly float min_W_i_on;
        /// <summary>
        /// max value of local bias in OFF channels
        /// </summary>
        public readonly float max_W_i_off;
        /// <summary>
        /// min value of local bias in OFF channels
        /// </summary>
        public readonly float min_W_i_off;
        /// <summary>
        /// max value of highpass time constant in SFA mechanism
        /// </summary>
        public readonly int max_tau_hp;
        /// <summary>
        /// min value of highpass time constant in SFA mechanism
        /// </summary>
        public readonly int min_tau_hp;
        /// <summary>
        /// max value of lowpass time constant in spatiotemporal convolution
        /// </summary>
        public readonly int max_tau_lp;
        /// <summary>
        /// min value of lowpass time constant in spatiotemporal convolution
        /// </summary>
        public readonly int min_tau_lp;
        /// <summary>
        /// Coefficient in contrast normalisation field
        /// </summary>
        protected readonly float coe_cn;
        /// <summary>
        /// Width of convolution kernel in contrast normalisation field
        /// </summary>
        protected readonly int Wcn;
        /// <summary>
        /// Standard deviation in contrast normalisation field
        /// </summary>
        protected readonly float std_cn;
        /// <summary>
        /// Delay coefficient in Medulla layer
        /// </summary>
        protected readonly float delay_lp;
        /// <summary>
        /// Time constant in low-pass filtering of signal correlation in milliseconds
        /// </summary>
        protected readonly int tau_llp;
        /// <summary>
        /// Exponent of T4 neuron response
        /// </summary>
        protected readonly float exp_ON;
        /// <summary>
        /// Exponent of T5 neuron response
        /// </summary>
        protected readonly float exp_OFF;

        #endregion

        #region LGMD PROPERTY

        /// <summary>
        /// property of photoreceptor layer
        /// </summary>
        public int[, ,] PHOTOS
        {
            get { return photoreceptors; }
            set { photoreceptors = value; }
        }
        /// <summary>
        /// property of ON cells
        /// </summary>
        public float[, ,] ONs
        {
            get { return ons; }
            set { ons = value; }
        }
        /// <summary>
        /// property of OFF cells
        /// </summary>
        public float[, ,] OFFs
        {
            get { return offs; }
            set { offs = value; }
        }
        /*
        public float[,] ON_LP
        {
            get { return on_lp; }
            set { on_lp = value; }
        }
        public float[,] OFF_LP
        {
            get { return off_lp; }
            set { off_lp = value; }
        }
        */
        /// <summary>
        /// property of ON inhibition
        /// </summary>
        public float[,] ON_INH
        {
            get { return on_inh; }
            set { on_inh = value; }
        }
        /// <summary>
        /// property of OFF excitation
        /// </summary>
        public float[,] OFF_INH
        {
            get { return off_inh; }
            set { off_inh = value; }
        }
        /// <summary>
        /// property of summation cells
        /// </summary>
        public float[,] S_CELLS_ON
        {
            get { return scells_on; }
            set { scells_on = value; }
        }
        public float[,] S_CELLS_OFF
        {
            get { return scells_off; }
            set { scells_off = value; }
        }
        /// <summary>
        /// property of grouping cells
        /// </summary>
        public float[,] G_CELLS_ON
        {
            get { return gcells_on; }
            set { gcells_on = value; }
        }
        public float[,] G_CELLS_OFF
        {
            get { return gcells_off; }
            set { gcells_off = value; }
        }
        /// <summary>
        /// property of FFI value
        /// </summary>
        public float[] FFI
        {
            get { return ffi; }
            set { ffi = value; }
        }
        /*
        public float FFE
        {
            get { return ffe; }
            set { ffe = value; }
        }
        */
        /// <summary>
        /// property of neural membrane potential
        /// </summary>
        public float[] MP
        {
            get { return mp; }
            set { mp = value; }
        }
        /// <summary>
        /// property of sigmoid membrane potential
        /// </summary>
        public float[] SMP
        {
            get { return smp; }
            set { smp = value; }
        }
        /// <summary>
        /// property of spike frequency adaptation
        /// </summary>
        public float[] SFA
        {
            get { return sfa; }
            set { sfa = value; }
        }
        public float[] DIFF_SMP
        {
            get { return diff_smp; }
            set { diff_smp = value; }
        }
        /// <summary>
        /// property of spike
        /// </summary>
        public byte SPIKE
        {
            get { return spike; }
            set { spike = value; }
        }
        /// <summary>
        /// property of collision detection
        /// </summary>
        public byte COLLISION
        {
            get { return collision; }
            set { collision = value; }
        }
        /// <summary>
        /// property of local threshold in S layer
        /// </summary>
        public int TS
        {
            get { return this.Ts; }
            set { this.Ts = value; }
        }
        /// <summary>
        /// property of spiking threshold
        /// </summary>
        public float TSP
        {
            get { return this.Tsp; }
            set { this.Tsp = value; }
        }
        /// <summary>
        /// property of FFI threshold
        /// </summary>
        public int TFFI
        {
            get { return this.Tffi; }
            set { this.Tffi = value; }
        }
        /// <summary>
        /// property of coefficient in sigmoid tranformation
        /// </summary>
        public float COE_SIG
        {
            get { return this.coe_sig; }
            set { this.coe_sig = value; }
        }
        /// <summary>
        /// property of inhibitory bias in ON channels
        /// </summary>
        public float W_I_ON
        {
            get { return this.W_i_on; }
            set { this.W_i_on = value; }
        }
        /// <summary>
        /// property of inhibitory bias in OFF channels
        /// </summary>
        public float W_I_OFF
        {
            get { return this.W_i_off; }
            set { this.W_i_off = value; }
        }
        /// <summary>
        /// property of time constant in highpass
        /// </summary>
        public int TAU_HP
        {
            get { return this.tau_hp; }
            set { this.tau_hp = value; }
        }
        /// <summary>
        /// property of time constant in lowpass
        /// </summary>
        public int TAU_LP
        {
            get { return this.tau_lp[0]; }
            set { this.tau_lp[0] = value; }
        }

        #endregion

        #region GA PROPERTY

        /// <summary>
        /// property of searching parameters dictionary
        /// </summary>
        public Dictionary<string, float> ParamsDict
        {
            get { return paramsDict; }
            set { paramsDict = value; }
        }

        /// <summary>
        /// property of neuron activating timing
        /// </summary>
        public int ActivationTiming
        {
            get { return activationTiming; }
            set { activationTiming = value; }
        }

        /// <summary>
        /// property of fitness value
        /// </summary>
        public float Fitness
        {
            get { return fitness; }
            set { fitness = value; }
        }

        /// <summary>
        /// property of old generation checking
        /// </summary>
        public bool IsOldGeneration
        {
            get { return isOldGeneration; }
            set { isOldGeneration = value; }
        }

        /// <summary>
        /// property of failure of collision detection
        /// </summary>
        public int FCOL
        {
            get { return Fcol; }
            set { Fcol = value; }
        }

        /// <summary>
        /// property of wrong response to non-collision events
        /// </summary>
        public int FNON
        {
            get { return Fnon; }
            set { Fnon = value; }
        }

        /// <summary>
        /// property of best agent
        /// </summary>
        public bool IsBestAgent
        {
            get { return isBestAgent; }
            set { isBestAgent = value; }
        }

        /// <summary>
        /// property of parent selection
        /// </summary>
        public bool IsParent
        {
            get { return isParent; }
            set { isParent = value; }
        }

        #endregion

        #region CONSTRUCTOR

        /// <summary>
        /// Constructor
        /// </summary>
        public LGMDs()
        { }
        /// <summary>
        /// Parametric Constructor
        /// </summary>
        /// <param name="w"></param>
        /// <param name="h"></param>
        /// <param name="fps"></param>
        public LGMDs(int width /*frame width*/, int height /*frame height*/, int fps /*frames per second*/)
        {
            this.width = width;
            this.height = height;
            dc = 0.1f;
            //gene: W_i_on
            W_i_on = 0.4f;
            //gene: W_i_off
            W_i_off = 0.4f;
            //LGMD1
            W_on = 1;
            W_off = 1;
            W_onoff = 1;
            Ncell = width * height;
            //gene: coe_sig
            coe_sig = 0.005f;
            //gene: Ts
            Ts = 30;
            //gene: Tffi
            Tffi = 20;
            //gene: Tsp
            Tsp = 0.68f;
            sp_step = 0.1f;
            Tsf = 0.0001f;
            Nsp = 0;
            Nts = 5;
            Np = 1;
            clip_point = 0.1f;
            time_interval = 1000 / fps; //ms
            Ksp = 4;
            Wcn = 11;
            std_cn = 5;
            coe_cn = 3;
            Delta_C = 0.01f;
            Cw = 4;
            Cde = 10.0f;
            Tde = 1;
            tau_llp = 30;
            delay_lp = time_interval / (tau_llp + time_interval);
            exp_ON = 0.5f;
            exp_OFF = 0.5f;
            sd = 4;
            /*********************attention****************************/
            //temporal parameters
            //gene: tau_hp
            tau_hp = 800; //ms
            hp_delay = tau_hp / (time_interval + tau_hp);
            //gene: tau_lp
            tau_lp = new int[] { 30, 60, 90 }; //ms
            lp_delay = new float[tau_lp.Length];
            for (int i = 0; i < tau_lp.Length; i++)
            {
                lp_delay[i] = time_interval / (time_interval + tau_lp[i]);
            }
            /*********************attention****************************/
            //convolving matrix init
            this.W_l = new float[2 * Np + 1, 2 * Np + 1];
            this.W_g = new float[2 * Np + 1, 2 * Np + 1];
            localKernel(ref W_l);
            groupKernel(ref W_g);
            //layers inits
            photoreceptors = new int[height, width, 2];
            ons = new float[height, width, 2];
            offs = new float[height, width, 2];
            ONs_Compressed = new float[height, width, 2];
            OFFs_Compressed = new float[height, width, 2];
            ONs_Contrast = new float[height, width, 2];
            OFFs_Contrast = new float[height, width, 2];
            Kernel_Constrast = new float[Wcn, Wcn];
            on_inh = new float[height, width];
            off_inh = new float[height, width];
            scells_on = new float[height, width];
            scells_off = new float[height, width];
            gcells_on = new float[height, width];
            gcells_off = new float[height, width];
            ONs_delay = new float[height, width];
            OFFs_delay = new float[height, width];
            T4s = new float[height, width, 2, 4];
            T5s = new float[height, width, 2, 4];
            LPTCs = new float[4];
            LPis = new float[4];
            ffi = new float[2];
            mp = new float[2];
            smp = new float[2];
            sfa = new float[2];
            diff_smp = new float[2];
            spike = 0;
            collision = 0;
            MakeGaussian(std_cn, Wcn, ref Kernel_Constrast);
#if BANDPASS
            sd = 4;
            esd = 0.5f * sd;
            isd = sd;
            //widths are always set to odd numbers
            inh_gauss_width = 2 * sd + 1;
            exc_gauss_width = sd + 1;
            inh_gauss_kernel = new float[inh_gauss_width, inh_gauss_width];
            exc_gauss_kernel = new float[exc_gauss_width, exc_gauss_width];
            MakeGaussian(esd, exc_gauss_width, ref exc_gauss_kernel);
            MakeGaussian(isd, inh_gauss_width, ref inh_gauss_kernel);
#endif

            //attention for GA parameters initialisation
            paramsDict = new Dictionary<string, float>();
            isOldGeneration = false;
            fitness = 0;
            activationTiming = 0;
            isBestAgent = false;
            isParent = false;
            Fcol = 0;
            Fnon = 0;
            max_Ts = 50;
            min_Ts = 10;
            max_Tsp = 0.91f;
            min_Tsp = 0.65f;
            max_Tffi = 20;
            min_Tffi = 5;
            max_Csig = 2.0f;
            min_Csig = 0.1f;
            max_tau_hp = 1300;
            min_tau_hp = 300;
            max_tau_lp = 50;
            min_tau_lp = 10;
            max_W_i_off = 2.0f;
            min_W_i_off = 0.5f;
            max_W_i_on = 1.0f;
            min_W_i_on = 0.1f;
            paramsDict.Add("Tsp", this.Tsp);
            paramsDict.Add("Tffi", this.Tffi);
            paramsDict.Add("Ts", this.Ts);
            paramsDict.Add("coe_sig", this.coe_sig);
            paramsDict.Add("tau_hp", this.tau_hp);
            paramsDict.Add("tau_lp", this.tau_lp[0]);
            paramsDict.Add("W_i_on", this.W_i_on);
            paramsDict.Add("W_i_off", this.W_i_off);

            Console.WriteLine("LGMDs Neuron Model Parameters Settings Ready........\n");
        }

        #endregion

        #region METHODS
        /// <summary>
        /// Make gauss filter kernel
        /// </summary>
        /// <param name="sigma"></param>
        /// <param name="gauss_width"></param>
        /// <param name="gauss_kernel"></param>
        public void MakeGaussian(float sigma, int gauss_width, ref float[,] gauss_kernel)
        {
            //coordinate of center point
            int center_x = gauss_width / 2;
            int center_y = gauss_width / 2;

            float gaussian; //record different gaussian value according to different sigma
            float distance;
            float sum = 0;

            for (int i = 0; i < gauss_width; i++)
            {
                for (int j = 0; j < gauss_width; j++)
                {
                    distance = (center_x - i) * (center_x - i) + (center_y - j) * (center_y - j);
                    gaussian = (float)(Math.Exp((0 - distance) / (2 * sigma * sigma)) / (2 * Math.PI * sigma * sigma));
                    sum += gaussian;
                    gauss_kernel[i, j] = gaussian;
                }
            }

            for (int i = 0; i < gauss_width; i++)
            {
                for (int j = 0; j < gauss_width; j++)
                {
                    gauss_kernel[i, j] /= sum;
                }
            }
        }
        /// <summary>
        /// Gaussian convolution
        /// </summary>
        /// <param name="input"></param>
        /// <param name="gauss_width"></param>
        /// <param name="gauss_kernel"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        protected int[,] Average_Filter(int[, ,] input, int gauss_width, float[,] gauss_kernel, int t)
        {
            int[,] output = new int[height, width];
            int tmp = 0;
            //kernel radius
            int k_radius = gauss_width / 2;
            int r, c;
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    for (int i = -k_radius; i < k_radius + 1; i++)
                    {
                        for (int j = -k_radius; j < k_radius + 1; j++)
                        {
                            r = y + i;
                            c = x + j;
                            //if exceeding range, let it equal to near pixel
                            while (r < 0)
                            { r++; }
                            while (r >= height)
                            { r--; }
                            while (c < 0)
                            { c++; }
                            while (c >= width)
                            { c--; }
                            //************************
                            tmp = (int)(input[r, c, t] * gauss_kernel[i + k_radius, j + k_radius]);
                            output[y, x] += tmp;
                        }
                    }
                }
            }
            return output;
        }
        /// <summary>
        /// Subtraction of DoGs with polarity selectivity
        /// </summary>
        /// <param name="first_image"></param>
        /// <param name="second_image"></param>
        /// <returns></returns>
        protected int[,] Sub_Images(int[,] first_image, int[,] second_image)
        {
            int[,] result_image = new int[height, width];

            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    result_image[i, j] = first_image[i, j] - second_image[i, j];

                    if (first_image[i, j] >= 0 && second_image[i, j] >= 0)
                    {
                        result_image[i, j] = Math.Abs(result_image[i, j]);
                    }
                    else if (first_image[i, j] <= 0 && second_image[i, j] <= 0)
                    {
                        if (result_image[i, j] > 0)
                            result_image[i, j] = -result_image[i, j];
                    }

                }
            }
            return result_image;
        }
        /// <summary>
        /// Difference of Gaussians algorithm
        /// </summary>
        /// <param name="input_image"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        protected int[,] DOG(int[, ,] input_image, int t)
        {
            //MakeGaussian(Esd, exc_gauss_width, ref exc_gauss_kernel);
            //MakeGaussian(Isd, inh_gauss_width, ref inh_gauss_kernel);
            //first convolution
            int[,] tmp1 = Average_Filter(input_image, exc_gauss_width, exc_gauss_kernel, t);
            //second convolution
            int[,] tmp2 = Average_Filter(input_image, inh_gauss_width, inh_gauss_kernel, t);
            //difference of gaussians
            return Sub_Images(tmp1, tmp2);
        }
        /// <summary>
        /// Make convolution kenerl in the dual-pathways
        /// </summary>
        /// <param name="mat"></param>
        protected void localKernel(ref float[,] mat)
        {
            for (int i = -1; i < Np + 1; i++)
            {
                for (int j = -1; j < Np + 1; j++)
                {
                    if (i == 0 && j == 0)   //centre
                        mat[i + 1, j + 1] = 0;
                    else if (i == 0 || j == 0)  //nearest
                        mat[i + 1, j + 1] = 0.25f;
                    else   //diagonal
                        mat[i + 1, j + 1] = 0.125f;
                }
            }
        }
        /// <summary>
        /// Make convolution kernel in the grouping layer
        /// </summary>
        /// <param name="mat"></param>
        protected void groupKernel(ref float[,] mat)
        {
            for (int i = 0; i < 2 * Np + 1; i++)
            {
                for (int j = 0; j < 2 * Np + 1; j++)
                {
                    mat[i, j] = 1 / 9f;
                }
            }
        }
        /// <summary>
        /// The simple version of first-order high-pass filter with no residual visual information
        /// </summary>
        /// <param name="pre_input"></param>
        /// <param name="cur_input"></param>
        /// <returns></returns>
        protected int HighpassFilter(byte pre_input, byte cur_input)
        {
            return cur_input - pre_input;
        }
        /// <summary>
        /// Spike-Frequency Adaptation (SFA) mechanism
        /// </summary>
        /// <param name="pre_sfa"></param>
        /// <param name="pre_mp"></param>
        /// <param name="cur_mp"></param>
        /// <returns></returns>
        protected float SFA_HPF(float pre_sfa, float pre_mp, float cur_mp)
        {
            float diff_mp = cur_mp - pre_mp;
            if (diff_mp <= Tsf)
            {
                float tmp_mp = hp_delay * (pre_sfa + diff_mp);
                if (tmp_mp < 0.5f)
                    return 0.5f;
                else
                    return tmp_mp;
            }
            else
            {
                float tmp_mp = hp_delay * cur_mp;
                if (tmp_mp < 0.5f)
                    return 0.5f;
                else
                    return tmp_mp;
            }
        }
        /// <summary>
        /// Selective SFA depending on the changing profile of membrane potential
        /// </summary>
        /// <param name="pre_diff"></param>
        /// <param name="cur_diff"></param>
        /// <param name="pre_sfa"></param>
        /// <param name="cur_mp"></param>
        /// <returns></returns>
        protected float SFA_Profile(float pre_diff, float cur_diff, float pre_sfa, float cur_mp)
        {
            float tmp_mp;
            if (cur_diff <= Tsf)  //first-derivative -> negative -> fast adaptation
            {
                tmp_mp = hp_delay * (pre_sfa + cur_diff);
            }
            else if (cur_diff > Tsf && cur_diff <= pre_diff)   //first-derivative -> positive & second-derivative -> negative -> little adaptation
            {
                tmp_mp = hp_delay * (pre_sfa + cur_diff);
            }
            else   //second-derivative -> positive -> least adaptation
            {
                tmp_mp = hp_delay * cur_mp;
            }
            if (tmp_mp < 0.5f)
            {
                tmp_mp = 0.5f;
            }
            return tmp_mp;
        }
        /// <summary>
        /// Half-wave rectifying in terms of onset response
        /// </summary>
        /// <param name="pre_output"></param>
        /// <param name="cur_input"></param>
        /// <returns></returns>
        protected float HRplusDC_ON(float pre_output, int cur_input)
        {
            if (cur_input >= clip_point)
                return cur_input + dc * pre_output;
            else
                return dc * pre_output;
        }
        /// <summary>
        /// Half-wave rectifying in terms of onset response
        /// </summary>
        /// <param name="pre_output"></param>
        /// <param name="cur_input"></param>
        /// <returns></returns>
        protected float HRplusDC_ON(float pre_output, float cur_input)
        {
            if (cur_input >= clip_point)
                return cur_input + dc * pre_output;
            else
                return dc * pre_output;
        }
        /// <summary>
        /// Half-wave rectifying in terms of offset response
        /// </summary>
        /// <param name="pre_output"></param>
        /// <param name="cur_input"></param>
        /// <returns></returns>
        protected float HRplusDC_OFF(float pre_output, int cur_input)
        {
            if (cur_input < clip_point)
                return Math.Abs(cur_input) + dc * pre_output;
            else
                return dc * pre_output;
        }
        /// <summary>
        /// Half-wave rectifying in terms of offset response
        /// </summary>
        /// <param name="pre_output"></param>
        /// <param name="cur_input"></param>
        /// <returns></returns>
        protected float HRplusDC_OFF(float pre_output, float cur_input)
        {
            if (cur_input < clip_point)
                return Math.Abs(cur_input) + dc * pre_output;
            else
                return dc * pre_output;
        }
        /// <summary>
        /// The first-order low-pass filter
        /// </summary>
        /// <param name="cur_input"></param>
        /// <param name="pre_input"></param>
        /// <param name="lp_t"></param>
        /// <returns></returns>
        protected float LowpassFilter(float cur_input, float pre_input, float lp_t)
        {
            return lp_t * cur_input + (1 - lp_t) * pre_input;
        }
        /// <summary>
        /// Convolution
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="matrix"></param>
        /// <param name="kernel"></param>
        /// <returns></returns>
        protected float Convolving(int x, int y, float[,] matrix, float[,] kernel)
        {
            float tmp = 0;
            int r, c;
            for (int i = -Np; i < Np + 1; i++)
            {
                r = x + i;
                while (r < 0)
                    r += 1;
                while (r >= height)
                    r -= 1;
                for (int j = -Np; j < Np + 1; j++)
                {
                    c = y + j;
                    while (c < 0)
                        c += 1;
                    while (c >= width)
                        c -= 1;
                    tmp += matrix[r, c] * kernel[i + Np, j + Np];
                }
            }
            return tmp;
        }
        /// <summary>
        /// Convolution with temporal information
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="inputMatrix"></param>
        /// <param name="kernel"></param>
        /// <param name="cur_frame"></param>
        /// <param name="pre_frame"></param>
        /// <returns></returns>
        protected float Convolve_ST(int x, int y, float[, ,] inputMatrix, float[,] kernel, int cur_frame, int pre_frame)
        {
            float tmp = 0;
            int r, c;
            float lp;
            for (int i = -Np; i < Np + 1; i++)
            {
                r = x + i;
                while (r < 0)
                    r += 1;
                while (r >= height)
                    r -= 1;
                for (int j = -Np; j < Np + 1; j++)
                {
                    c = y + j;
                    while (c < 0)
                        c += 1;
                    while (c >= width)
                        c -= 1;
                    if (i == 0 && j == 0)
                        lp = 0;
                    else if (i == 0 || j == 0)
                        lp = LowpassFilter(inputMatrix[r, c, cur_frame], inputMatrix[r, c, pre_frame], lp_delay[0]);
                    else
                        lp = LowpassFilter(inputMatrix[r, c, cur_frame], inputMatrix[r, c, pre_frame], lp_delay[1]);
                    tmp += lp * kernel[i + Np, j + Np];
                }
            }
            return tmp;
        }
        /// <summary>
        /// Lowpass for time delay
        /// </summary>
        /// <param name="cur_in">current input signal</param>
        /// <param name="pre_out">previous output signal</param>
        /// <param name="a">delay coefficient</param>
        /// <returns></returns>
        protected float Lowpass(float cur_in, float pre_out, float a)
        {
            return a * cur_in + (1 - a) * pre_out;
        }
        /// <summary>
        /// Hyberbolic Tangent function
        /// </summary>
        /// <param name="input">input node value</param>
        /// <returns></returns>
        protected float TanH(float input)
        {
            return ((float)Math.Exp(2 * input) - 1) / ((float)Math.Exp(2 * input) + 1);
        }
        /// <summary>
        /// Contrast normalization
        /// </summary>
        /// <param name="kernel">convolution kernel</param>
        /// <param name="inSignal">input signal matrix</param>
        /// <param name="x">abscissa</param>
        /// <param name="y">ordinate</param>
        /// <param name="t">time</param>
        /// <returns></returns>
        protected float normaliseContrast(float[,] kernel, float[, ,] inSignal, int x, int y, int t)
        {
            float tmp = 0;
            int np = kernel.GetLength(0) / 2;
            int r, c;
            for (int i = -np; i < np + 1; i++)
            {
                r = x + i;
                while (r < 0)
                    r += 1;
                while (r >= height)
                    r -= 1;
                for (int j = -np; j < np + 1; j++)
                {
                    c = y + j;
                    while (c < 0)
                        c += 1;
                    while (c >= width)
                        c -= 1;
                    tmp += inSignal[r, c, t] * kernel[i + np, j + np];
                }
            }
            // Activation
            return TanH(inSignal[x, y, t] / (coe_cn + tmp));
        }
        /// <summary>
        /// Contrast vision from Medulla layer (Inhibition)
        /// </summary>
        /// <param name="x">abscissa</param>
        /// <param name="y">ordinate</param>
        /// <param name="matrix">input signal matrix</param>
        /// <param name="t">time</param>
        /// <returns></returns>
        protected float ContrastVision(int x, int y, float[, ,] matrix, int t)
        {
            float tmp = 0;
            int r, c;
            for (int i = -1; i < 1 + 1; i++)
            {
                r = x + i;
                while (r < 0)
                    r += 1;
                while (r >= height)
                    r -= 1;
                for (int j = -1; j < 1 + 1; j++)
                {
                    c = y + j;
                    while (c < 0)
                        c += 1;
                    while (c >= width)
                        c -= 1;
                    tmp += matrix[r, c, t];
                }
            }
            return (float)Math.Abs(matrix[x, y, t] - (tmp - matrix[x, y, t]) / 8);
        }
        /// <summary>
        /// Local summation of ON cells
        /// </summary>
        /// <param name="exc"></param>
        /// <param name="inh"></param>
        /// <returns></returns>
        protected float Scell_ON(float exc, float inh)
        {
            float value = exc - W_i_on * inh;
            return value;

        }
        /// <summary>
        /// Local summation of OFF cells
        /// </summary>
        /// <param name="exc"></param>
        /// <param name="inh"></param>
        /// <returns></returns>
        protected float Scell_OFF(float exc, float inh)
        {
            float value = exc - W_i_off * inh;
            return value;
        }
        /// <summary>
        /// Local interactions of excitations of ON and OFF cells
        /// </summary>
        /// <param name="on_exc"></param>
        /// <param name="off_exc"></param>
        /// <returns></returns>
        protected float SupralinearSummation(float on_exc, float off_exc)
        {
            float tmp = W_on * on_exc + W_off * off_exc + W_onoff * on_exc * off_exc;
            return tmp;
        }
        /// <summary>
        /// Feed-forward inhibition pathway processing
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="t"></param>
        /// <returns></returns>
        protected float Calc_FFI(int[, ,] matrix, int t)
        {
            float tmp = 0;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    tmp += Math.Abs(matrix[i, j, t]);
                }
            }
            return tmp / Ncell;
        }
        /// <summary>
        /// Logarithmic membrane potential calculation
        /// </summary>
        /// <param name="ffe"></param>
        /// <param name="ffi"></param>
        /// <returns></returns>
        protected float Calc_MP(float ffe, float ffi)
        {
            if (ffe == 0 || ffi == 0)
                return 0;
            else
                return (float)(Math.Log(ffe) - Math.Log(/*coe_ffi **/ ffi));
            //return ffe / (coe_ffi * ffi);
        }
        /// <summary>
        /// Sigmoid transformation of membrane potential
        /// </summary>
        /// <param name="Kf"></param>
        /// <returns></returns>
        protected float SigmoidTransfer(float Kf)
        {
            return (float)Math.Pow(1 + Math.Exp(-Kf * Math.Pow(Ncell * coe_sig, -1)), -1);
        }
        /// <summary>
        /// Spiking mechanism
        /// </summary>
        /// <param name="smp"></param>
        /// <returns></returns>
        protected byte SpikingMechanism(float mp)
        {
            if (mp < Tsp)
            {
                Nsp = 0;
                return 0;
            }
            else if (mp >= Tsp && mp < Tsp + sp_step)
            {
                Nsp++;
                return 1;
            }
            else
            {
                Nsp += 2;
                return 2;
            }
        }
        /// <summary>
        /// Spiking mechanism via exponentially mapping
        /// </summary>
        /// <param name="sfa"></param>
        /// <returns></returns>
        protected byte Spiking(float sfa)
        {
            byte spi = (byte)Math.Floor(Math.Exp(Ksp * (sfa - Tsp)));
            if (spi == 0)
                Nsp = 0;
            else
                Nsp += spi;
            return spi;
        }
        /// <summary>
        /// Detect potential collisions
        /// </summary>
        /// <returns></returns>
        protected byte loomingDetecting()
        {
            if (Nsp >= Nts)
                return 1;
            else
                return 0;
        }
        /// <summary>
        /// A scale computation in G layer
        /// </summary>
        /// <param name="cur_frame"></param>
        /// <returns></returns>
        protected float ScaleON(int cur_t)
        {
            float max = 0;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    if (max < Math.Abs(G_CELLS_ON[i, j]))
                        max = Math.Abs(G_CELLS_ON[i, j]);
                }
            }
            return (float)(Delta_C + max * Math.Pow(Cw, -1));
        }
        protected float ScaleOFF(int cur_t)
        {
            float max = 0;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    if (max < Math.Abs(G_CELLS_OFF[i, j]))
                        max = Math.Abs(G_CELLS_OFF[i, j]);
                }
            }
            return (float)(Delta_C + max * Math.Pow(Cw, -1));
        }
        /// <summary>
        /// Computation and thresholding of each Grouping cell
        /// </summary>
        /// <param name="scellvalue"></param>
        /// <param name="ce"></param>
        /// <param name="w"></param>
        /// <returns></returns>
        protected float gCellValue(float scellvalue, float ce, float w)
        {
            float value = scellvalue * ce * (float)Math.Pow(w, -1);
            if (value * Cde >= Tde)
                return value;
            else
                return 0;
        }
        /// <summary>
        /// Diverging triple correlation
        /// </summary>
        /// <param name="cur_p">current position signal</param>
        /// <param name="cur_p_d">currrent position+distance signal</param>
        /// <param name="pre_p">previous position signal</param>
        /// <param name="pre_p_d">previous position+distance signal</param>
        /// <returns></returns>
        protected float Div_TripleCorrelation(float cur_p, float cur_p_d, float pre_p, float pre_p_d)
        {
            return pre_p * cur_p_d - cur_p * pre_p_d;
        }
        /// <summary>
        /// Polarity channels summation each with fractional order
        /// </summary>
        /// <param name="on_exc">ON channel excitation</param>
        /// <param name="off_exc">OFF channel excitation</param>
        /// <param name="on_exp">exponent of ON channel output</param>
        /// <param name="off_exp">exponent of OFF channel output</param>
        /// <returns></returns>
        protected float PolarityChannelSummation(float on_exc, float off_exc, float on_exp, float off_exp)
        {
            return (float)(Math.Pow(on_exc, on_exp) + Math.Pow(off_exc, off_exp));
        }
        /// <summary>
        /// Rectified Linear Unit (ReLU)
        /// </summary>
        /// <param name="input">input node value</param>
        /// <returns></returns>
        protected float ReLU(float input)
        {
            if (input < 0)
                return 0;
            else
                return input;
        }
        /// <summary>
        /// Sigmoid Activation
        /// </summary>
        /// <param name="input">input node value</param>
        /// <param name="a">scale coefficient</param>
        /// <returns></returns>
        protected float Sigmoid(float input, float a)
        {
            return (float)Math.Pow(1 + Math.Exp(-Math.Abs(input) * Math.Pow(a, -1)), -1);
        }
        #endregion

        #region LGMD DUAL-PATHWAYS MODEL PROCESSING
        /// <summary>
        /// The signal processing of general LGMDs neural network
        /// </summary>
        /// <param name="img1">first input image</param>
        /// <param name="img2">second input image</param>
        /// <param name="t">current time point</param>
        public void LGMDSProcessing(byte[, ,] img1 /*first input image*/, byte[, ,] img2 /*second input image*/, int t /*time*/, string filename /*file name*/, bool flag)
        {
            int cur_frame = t % 2;
            int pre_frame = (t - 1) % 2;
            float tmp_ffi = 0.0f;
            float tmp_ffi2 = 0.0f;
            float tmp_ffi3 = 0.0f;
            float sca_on = 0.0f;
            float sca_off = 0.0f;
            float tmp_sum_max = 0.0f;
            float tmp_sum_min = 0.0f;
            if (true == flag)
            {
                Console.WriteLine("******************************************************************************");
                Console.WriteLine("******************************************************************************");
                Array.Clear(PHOTOS, 0, PHOTOS.Length);
                Array.Clear(FFI, 0, FFI.Length);
                Array.Clear(MP, 0, MP.Length);
                Array.Clear(SMP, 0, SMP.Length);
                Array.Clear(SFA, 0, SFA.Length);
                Array.Clear(ONs, 0, ONs.Length);
                Array.Clear(OFFs, 0, OFFs.Length);
                Array.Clear(ONs_Compressed, 0, ONs_Compressed.Length);
                Array.Clear(OFFs_Compressed, 0, OFFs_Compressed.Length);
                Array.Clear(ONs_Contrast, 0, ONs_Contrast.Length);
                Array.Clear(OFFs_Contrast, 0, OFFs_Contrast.Length);
                Array.Clear(ON_INH, 0, ON_INH.Length);
                Array.Clear(OFF_INH, 0, OFF_INH.Length);
                Array.Clear(S_CELLS_ON, 0, S_CELLS_ON.Length);
                Array.Clear(S_CELLS_OFF, 0, S_CELLS_OFF.Length);
                Array.Clear(G_CELLS_ON, 0, G_CELLS_ON.Length);
                Array.Clear(G_CELLS_OFF, 0, G_CELLS_OFF.Length);
            }

            // Retina neural layer
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    PHOTOS[y, x, cur_frame] = HighpassFilter(img1[y, x, 0], img2[y, x, 0]);  // Corresponding formulas 2 and 3
                    tmp_ffi += Math.Abs(PHOTOS[y, x, cur_frame]);                            // Corresponding formula 4
                }
            }
			// FFI biological mechanism
            FFI[cur_frame] = tmp_ffi / Ncell;                                                // Corresponding formula 4
            FFI[cur_frame] = LowpassFilter(FFI[cur_frame], FFI[pre_frame], lp_delay[2]);     // Corresponding formula 5
            if (FFI[cur_frame] >= Tffi)                                                      // Corresponding formula 5
            {
                MP[cur_frame] = 0;
                SMP[cur_frame] = 0.5f;
                SFA[cur_frame] = 0.5f;
            }
            else
            {
				// Lamina neural layer
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        ONs[y, x, cur_frame] = HRplusDC_ON(ONs[y, x, pre_frame], PHOTOS[y, x, cur_frame]);     // Corresponding formula 6
                        OFFs[y, x, cur_frame] = HRplusDC_OFF(OFFs[y, x, pre_frame], PHOTOS[y, x, cur_frame]);  // Corresponding formula 6
                    }
                }

                // Medulla neural layer
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        ONs_Compressed[y, x, cur_frame] = normaliseContrast(Kernel_Constrast, ONs, y, x, cur_frame);    // Corresponding formulas 7 8 and 9
                        OFFs_Compressed[y, x, cur_frame] = normaliseContrast(Kernel_Constrast, OFFs, y, x, cur_frame);  // Corresponding formulas 7 8 and 9
                    }
                }

                for (int y = sd; y < height - sd; y++)
                {
                    for (int x = sd; x < width - sd; x++)
                    {
                        ONs_Contrast[y, x, cur_frame] = ContrastVision(y, x, ONs_Compressed, cur_frame);    // Corresponding formulas 10 and 11
                        OFFs_Contrast[y, x, cur_frame] = ContrastVision(y, x, OFFs_Compressed, cur_frame);  // Corresponding formulas 10 and 11
                    }
                }

                float tmp_sum = 0, tmp_sum_on = 0, tmp_sum_off = 0;
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        ON_INH[y, x] = Convolve_ST(y, x, ONs_Compressed, W_l, cur_frame, pre_frame);      // Corresponding formulas 12 and 13
                        OFF_INH[y, x] = Convolve_ST(y, x, OFFs_Compressed, W_l, cur_frame, pre_frame);    // Corresponding formulas 12 and 13
                        S_CELLS_ON[y, x] = Scell_ON(ONs_Compressed[y, x, cur_frame], ON_INH[y, x]);       // Corresponding formula 14
                        S_CELLS_OFF[y, x] = Scell_OFF(OFFs_Compressed[y, x, cur_frame], OFF_INH[y, x]);   // Corresponding formula 14
                    }
                }

                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        G_CELLS_ON[y, x] = Convolving(y, x, S_CELLS_ON, W_g);    // Corresponding formulas 15 and 16
                        G_CELLS_OFF[y, x] = Convolving(y, x, S_CELLS_OFF, W_g);  // Corresponding formulas 15 and 16
                    }
                }
                sca_on = ScaleON(cur_frame);    // Corresponding formula 18
                sca_off = ScaleOFF(cur_frame);  // Corresponding formula 18
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        G_CELLS_ON[y, x] = gCellValue(S_CELLS_ON[y, x], G_CELLS_ON[y, x], sca_on);      // Corresponding formulas 17 and 19
                        G_CELLS_OFF[y, x] = gCellValue(S_CELLS_OFF[y, x], G_CELLS_OFF[y, x], sca_off);  // Corresponding formulas 17 and 19
                        G_CELLS_ON[y, x] = G_CELLS_ON[y, x] - ONs_Contrast[y, x, cur_frame];            // Corresponding formula 20
                        G_CELLS_OFF[y, x] = G_CELLS_OFF[y, x] - OFFs_Contrast[y, x, cur_frame];         // Corresponding formula 20
						// Lobula complex neural layer
                        tmp_sum_on += G_CELLS_ON[y, x];                                                 // Corresponding formula 21
                        tmp_sum_off += G_CELLS_OFF[y, x];                                               // Corresponding formula 21
                    }
                }
                tmp_sum_on = Math.Abs(tmp_sum_on);
                tmp_sum_off = Math.Abs(tmp_sum_off);
                Console.WriteLine("Frame = {0} tmp_sum_on = {1:F} tmp_sum_off = {2:F}", t, tmp_sum_on, tmp_sum_off);
				
				// Lobula complex neural layer
                if (tmp_sum_on >= tmp_sum_off)   // Corresponding formula 22
                {
                    tmp_sum_max = tmp_sum_on;
                    tmp_sum_min = tmp_sum_off;
                }
                else   
                {
                    tmp_sum_max = tmp_sum_off;
                    tmp_sum_min = tmp_sum_on;
                }
                Console.WriteLine("Frame = {0} tmp_sum_max = {1:F} tmp_sum_min = {2:F}", t, tmp_sum_max, tmp_sum_min);


                if (tmp_sum_min <= 1.0f)      // Corresponding formula 23
                    tmp_sum = tmp_sum_max - tmp_sum_min;
                else
                    tmp_sum = tmp_sum_max / tmp_sum_min - 1.0f;

                MP[cur_frame] = tmp_sum;
                SMP[cur_frame] = SigmoidTransfer(MP[cur_frame]);  

                // Print to Console
                Console.WriteLine("Frame = {0} MP = {1:F} SMP = {2:F} FFI = {3:F}", t, MP[cur_frame], SMP[cur_frame], FFI[cur_frame]);

                // Write characters to file
                string showData = Convert.ToString(t);
                showData = showData + "_" + Convert.ToString(MP[cur_frame]) + "_" + Convert.ToString(SMP[cur_frame]);
                string str = "";
                str = @"..\..\..\Results\ImageSource\" + filename + ".txt";
                using (System.IO.StreamWriter file = new System.IO.StreamWriter(str, true))
                {
                    file.WriteLine(showData);
                    file.Close();
                }
            }
        }

        /// <summary>
        /// Exchanging searching parameters to object domain (LGMD dual pathways model)
        /// </summary>
        public void LGMDD_searchingParametersExchanging()
        {
            Tffi = (int)paramsDict["Tffi"];
            Ts = (int)paramsDict["Ts"];
            Tsp = paramsDict["Tsp"];
            coe_sig = paramsDict["coe_sig"];
            W_i_off = paramsDict["W_i_off"];
            W_i_on = paramsDict["W_i_on"];
            tau_hp = (int)paramsDict["tau_hp"];
            hp_delay = tau_hp / (time_interval + tau_hp);
            tau_lp[0] = (int)paramsDict["tau_lp"];
            lp_delay[0] = time_interval / (time_interval + tau_lp[0]);
            for (int i = 1; i < 2 * Np + 1; i++)
            {
                tau_lp[i] = tau_lp[0] + i * tau_lp[0];
                lp_delay[i] = time_interval / (time_interval + tau_lp[i]);
            }
        }

        /// <summary>
        /// LGMD NN dual-pathways model restore
        /// </summary>
        public void LGMDD_Restore()
        {
            for (int i = 0; i < ffi.Length; i++)
            {
                ffi[i] = 0;
                mp[i] = 0;
                smp[i] = 0;
                sfa[i] = 0;
            }
            photoreceptors = new int[height, width, 2];
            ons = new float[height, width, 2];
            offs = new float[height, width, 2];
        }

        /// <summary>
        /// Overwrite CompareTo method in IComparable
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public int CompareTo(Object obj)
        {
            if (this.fitness > ((LGMDs)obj).fitness)
                return -1;
            else if (this.fitness == ((LGMDs)obj).fitness)
                return 0;
            else
                return 1;
        }

        /// <summary>
        /// Overwrite Compare method in IComparer<LGMDs>
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public int Compare(LGMDs a, LGMDs b)
        {
            if (a.fitness < b.fitness)
                return -1;
            else if (a.fitness == b.fitness)
                return 0;
            else
                return 1;
        }

        #endregion
    }
}
