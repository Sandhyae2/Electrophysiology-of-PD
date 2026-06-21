import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import time
import scipy.io as sio
from datetime import datetime
import matplotlib.colors as mcolors

def MM_kin(substrate, Km, n):
    """Michaelis-Menten kinetics"""
    return (substrate**n) / ((Km**n) + (substrate**n))

def ConvertAPtoST(firings, step):
    """Convert action potential firings to spike times"""
    if len(firings) == 0:
        return np.array([])

    # Remove consecutive duplicates
    unique_firings = []
    if len(firings) > 0:
        unique_firings.append(firings[0])
        for i in range(1, len(firings)):
            if firings[i][0] != firings[i-1][0]:
                unique_firings.append(firings[i])

    return np.array(unique_firings)

def SNcATPapopNM(dur, gl, mt, Istim, filename, delay_ms=500,stim_duration_ms=1000,stim_amp_pA=50,pulse_width_ms=10,stim_freq_hz=20):
    """
    Substantia Nigra pars compacta model - soma+terminal+ATP+Apoptosis

    Parameters:
    dur: simulation duration (ms)
    gl: glycolysis factor
    mt: mitochondrial factor
    filename: output filename
    """

    start_time = time.time()

    # Time parameters
    dt = 0.1
    tspan = np.arange(dt, dur + dt, dt)
    Ttime = len(tspan)

    # Initialize arrays
    cai_array = np.zeros(Ttime)
    atpused_array = np.zeros(Ttime)
    apop_array = np.zeros(Ttime)
    eda_array = np.zeros(Ttime)
    V_snc_array = np.zeros(Ttime)
    ros_mit_array = np.zeros(Ttime)
    cda_array = np.zeros(Ttime)
    vda_array = np.zeros(Ttime)
    phi_er = np.zeros(Ttime)
    calb_array = np.zeros(Ttime)
    cam_array = np.zeros(Ttime)
    phi_mt = np.zeros(Ttime)
    caer_array = np.zeros(Ttime)
    camt_array = np.zeros(Ttime)
    LDOPA_array = np.zeros(Ttime)
    ATP_array = np.zeros(Ttime)
    LAC_array = np.zeros(Ttime)
    PYR_array = np.zeros(Ttime)
    GAP_array = np.zeros(Ttime)
    GSH_array = np.zeros(Ttime)
    F6P_array = np.zeros(Ttime)
    F26P_array = np.zeros(Ttime)
    PCr_array = np.zeros(Ttime)
    NADPH_array = np.zeros(Ttime)
    ROS_array = np.zeros(Ttime)
    ASYN_array = np.zeros(Ttime)
    ASYNA_array = np.zeros(Ttime)
    ASYNT_array = np.zeros(Ttime)
    ASYNG_array = np.zeros(Ttime)
    LB_array = np.zeros(Ttime)
    nai_array = np.zeros(Ttime)
    ki_array = np.zeros(Ttime)
    ADP_array = np.zeros(Ttime)
    uADP_array = np.zeros(Ttime)
    

    V_id = np.zeros(Ttime)
    V_dp = np.zeros(Ttime)
    V_er = np.zeros(Ttime)
    V_rel = np.zeros(Ttime)
    V_pro = np.zeros(Ttime)
    Iext_array = np.zeros(Ttime)

    # Initial conditions
    V_sncinit = -49.42
    Ca_iinit = 0.000188
    Na_iinit = 4.6876
    K_iinit = 126.05893
    Calbinit = 0.0026
    Caminit = 0.0222
    m_calinit = 0.006271
    m_nainit = 0.0952
    h_nainit = 0.1848
    O_hcninit = 0.003
    m_kdrinit = 0.0932
    y_pcinit = 0.483
    y_nkinit = 0.6213
    Ca_erinit = 1.0 * 0.001
    Ca_mtinit = 0.1 * 0.001
    cdainit = 1e-4
    vdainit = 500
    edainit = 4e-6
    Iextinit = 0
    ATPusedinit = 0
    calinit = 1
    cai_calinit = 0
    cal_actinit = 0
    casp12init = 1
    cal_act_casp12init = 0
    casp12_actinit = 0
    casp9init = 1
    casp12_act_casp9init = 0
    casp9_actinit = 0
    casp3init = 1
    casp9_act_casp3init = 0
    casp3_actinit = 0
    apopinit = 0
    ROS_mitinit = 0
    PTP_mit_actinit = 0
    Cytc_mitinit = 1
    Cytcinit = 0
    Cytc_casp9init = 0
    IAPinit = 0
    casp9_act_IAPinit = 0
    casp3_act_IAPinit = 0
    NADPHinit = 250 * 0.001
    GSHinit = 2500 * 0.001
    F6Pinit = 0.175883476634895
    F26Pinit = 0.002191750879602
    GAPinit = 0.082507126186107
    PYRinit = 0.123910489378719
    LACinit = 0.598605032933119
    ATPinit = 2.395615876085214
    PCrinit = 18.044071098085976
    ROSinit = 1 * 0.001
    ASYNinit = 100 * 0.001
    ASYNAinit = 1 * 0.001
    ASYNTinit = 0.01 * 0.001
    ASYNGinit = 0 * 0.001
    LBinit = 0 * 0.001
    LDOPAinit = 36e-5
   

    # Initialize state variables
    V_snc = V_sncinit
    m_cal = m_calinit
    m_na = m_nainit
    h_na = h_nainit
    O_hcn = O_hcninit
    Calb = Calbinit
    Cam = Caminit
    y_nk = y_nkinit
    y_pc = y_pcinit
    m_kdr = m_kdrinit
    K_i = K_iinit
    Na_i = Na_iinit
    Ca_i = Ca_iinit
    Ca_er = Ca_erinit
    Ca_mt = Ca_mtinit

    cda = cdainit
    vda = vdainit
    eda = edainit
    Iexts = Iextinit
    ATPused = ATPusedinit
    cal = calinit
    cai_cal = cai_calinit
    cal_act = cal_actinit
    casp12 = casp12init
    cal_act_casp12 = cal_act_casp12init
    casp12_act = casp12_actinit
    casp9 = casp9init
    casp12_act_casp9 = casp12_act_casp9init
    casp9_act = casp9_actinit
    casp3 = casp3init
    casp9_act_casp3 = casp9_act_casp3init
    casp3_act = casp3_actinit
    apop = apopinit

    ROS_mit = ROS_mitinit
    PTP_mit_act = PTP_mit_actinit
    Cytc_mit = Cytc_mitinit
    Cytc = Cytcinit
    Cytc_casp9 = Cytc_casp9init
    IAP = IAPinit
    casp9_act_IAP = casp9_act_IAPinit
    casp3_act_IAP = casp3_act_IAPinit
    NADPH = NADPHinit
    GSH = GSHinit
    F6P = F6Pinit
    F26P = F26Pinit
    GAP = GAPinit
    PYR = PYRinit
    LAC = LACinit
    ATP = ATPinit
    PCr = PCrinit
    ROS = ROSinit
    ASYN = ASYNinit
    ASYNA = ASYNAinit
    ASYNT = ASYNTinit
    ASYNG = ASYNGinit
    LB = LBinit
    LDOPA = LDOPAinit
 

    # Simulation constants
    sim_mM = 1e-3
    sim_mM_msec = 1e-3 / 3.6e6
    sim_msec = 1 / 3.6e6
    sim_msec_mM = 1 / ((1e-3) * (3.6e6))

    # Parameters - Constants
    # Soma
    R = 8314.472  # mJ/mol.K gas constant
    T = 310.15  # K temp
    F = 96485.30929  # coulomb/mol faraday constant
    Ca_o = 1.8  # mM extracellular environment
    Na_o = 137  # mM
    K_o = 5.4  # mM
    vol_pmu = 5  # pl neuron volume
    fr_cyt = 0.5  # cytosolic fraction
    C_sp = 0.9e6  # pF/cm2 specific capacitance
    SVR_pmu = 1.6667e4  # 1/cm surface to volume ratio
    Calbtot = 0.005  # mM total calbindin buffer
    Camtot = 0.0235  # mM mitochondrial ca2+ capacity
    kcal_1 = 10  # 1/mM.ms
    kcal_2 = 2e-3  # 1/ms
    kcam_cd = 0.003  # 1/ms
    kcam_nd = 3  # 1/ms
    g_cal = 2101.2  # pA/mM calcium channel
    g_na = 907.68  # pA/mM sodium channel
    A_mna = 1.9651  # 1/ms na m gate
    B_mna = 0.0424  # 1/ms
    A_hna = 9.566e-5  # 1/ms na h gate
    B_hna = 0.5296  # 1/ms
    za_mna = 1.7127
    zb_mna = 1.5581
    za_hna = -2.4317
    zb_hna = -1.1868
    g_nalk = 0.0053  # pA/mM
    g_nahcn = 51.1  # pA/mM hcn channel
    cAMP = 1e-5  # mM
    g_ksk = 2.2515  # pA/mM sk potassium channel
    g_kdr = 31.237  # nS
    g_kir = 13.816  # nS
    k_2pc = 0.001  # 1/ms
    k_3pc = 0.001  # 1/ms
    k_4pc = 1  # 1/ms
    K_pco = 2  # mM
    k_pmca = 2.233
    dell = 0.35
    k_xm = 0.0166  # pA
    k_2nk = 0.04  # 1/ms
    k_3nk = 0.01  # 1/ms
    k_4nk = 0.165  # 1/ms
    K_nknai = 4.05  # mM
    K_nknao = 69.8  # mM
    K_nkki = 32.88  # mM
    K_nkko = 0.258  # mM
    k_nk = 1085.7  # pA
    V_tau = (R * T) / F
    vol_cyt = fr_cyt * vol_pmu
    P_c = 1.0 / (1.0 + cAMP / 0.001163)
    P_o = 1.0 / (1.0 + cAMP / 1.45e-05)
    P_E2Spc = 1.0 / (1.0 + K_pco / Ca_o)
    A_pmu = (SVR_pmu * vol_pmu * 0.001 * 0.001 * 0.001) / 1.0
    P_E2pc = 1.0 - P_E2Spc
    beta_pc = k_2pc * P_E2Spc + k_4pc * P_E2pc
    

    # CICR - ER
    rho_er = 0.01
    beta_er = 0.0025
    k_pump = 20.0 / 1000
    k_ch = 3000.0 / 1000
    K1 = 5.0 * 0.001
    k_leak = 0.05 / 1000

    # Mito
    rho_mt = 0.01
    beta_mt = 0.0025
    k_in = 0.0055 * 300 * 0.001 / 1000
    K2 = 0.8 * 0.001
    k_out = 125.0 / 1000
    k_m = 0.00625 / 1000
    K3 = 5.0 * 0.001

    # DA terminal
    krel = 0.031
    psi = 17.4391793
    nRRP = 5
    Veda_max = 1e-6
    Keda = 3e-5
    kcomt = 0.0083511
    vdao = 500
    vdas = 1e-2
    dara = 5e-5
    dars = 1e-2
    Vsynt_max = 250e-5
    Ksynt = 35e-4
    Ktyr = 46e-3
    TYR = 126e-3
    Kicda = 11e-2
    Kieda = 46e-3
    Vcda_max = 0.2 * 133.33e-6
    Kcda = 238e-4
    kmao = 0.00016

    # Apoptosis pathway
    k3f = 1
    k3b = 1 / 1e3
    k4f = 1 / 1e3
    k5f = 1
    k5b = 1 / 1e3
    k6f = 1 / 1e3
    k7f = 10
    k7b = 0.5 / 1e3
    k8f = 1 / 1e3
    k9f = 10
    k9b = 0.5 / 1e3
    k10f = 0.1 / 1e3
    k11f = 1

    k29f = 0.5
    k30f = 0.5
    k31f = 1
    k27f = 1
    k27b = 1 / 1e3
    k28f = 1 / 1e3
    k12f = 5
    k12b = 0.0035 / 1e3
    k13f = 5
    k13b = 0.0035 / 1e3

    Mit = 1
    Sig_ers = 0
    Sig_mts = 0
    PTP_mit = 1

    # Energy Metabolism
    GLCe = 1
    Vmax_hk = 2.5 / 1000
    Km_ATP_hk = 0.5
    KI_F6P = 0.068
    Vmax_pfk = 3.85 / 1000
    Km_ATP_pfk = 0.05
    Km_F6P_pfk = 0.18
    Km_F26P_pfk = 0.01
    Vmaxf_pfk2 = 2e-04 / 1000
    Vmaxr_pfk2 = 1.036e-04 / 1000
    Km_ATP_pfk2 = 0.05
    Km_F6P_pfk2 = 0.01
    Km_F26P_pfk2 = 0.0001
    Vmax_pk = 5.0 / 1000
    Km_ADP_pk = 0.005
    Km_GAP_pk = 0.4
    Vmax_op = 1.0 / 1000
    Km_ADP_op = 0.005
    Km_PYR_op = 0.5
    kf_ldh = 12.5 / 1000
    kr_ldh = 2.5355 / 1000
    kf_ck = 3.0 / 1000
    kr_ck = 1.26 / 1000
    PCr_tot = 20.0
    Vmax_ATPase = 0.9355 / 1000
    Km_ATP = 0.5
    Vlac_0 = 0.355 / 1000
    K_lac_eff = 0.71 / 1000
    K_lac = 0.641
    ANP = 2.51
    Q_adk = 0.92
    nATP = 0.4
    KI_ATP = 1.0
    nAMP = 0.5
    Ka_AMP = 0.05
    Kamp_pfk2 = 0.005
    nh_amp = 2
    beta_ldh_ros = 0.25
    Kldh_ros = 10 * sim_mM

    snc_firings = []

    kf_gr = 0.65 * sim_msec_mM
    kr_gr = 1.25e-3 * sim_msec_mM
    GSH_tot = 2500 * sim_mM
    NADPH_tot = 250 * sim_mM
    Vmax_ppp = 1.43e6 * sim_mM_msec
    Ki_nadph = 20

    # PD pathology pathways
    eta_op_max = 0.995
    beta_op_asyn = 0.08
    Kasyn = 8.5 * sim_mM
    Kros_cat = 235 * 1 * sim_msec
    Vros_ex = 0
    Kros_dopa = 1500 * sim_msec_mM
    Kros_dox = 0.27 * sim_msec
    Kasyn_syn = 50 * sim_mM_msec
    Kasyn_ox = 7e-5 * sim_msec_mM
    Kasyn_to = 0.5 * sim_msec
    Krasyn_agg = 7.5e-4 * sim_msec
    Kasyn_agg = 7.5 * sim_mM
    Ub_tot = 10.5 * sim_mM
    Kasyn_tag = 2.75e-7 * sim_msec_mM
    Krasyn_prt = 7.5e-4 * sim_msec
    Kasyn_prt = 5 * sim_mM
    beta_asyn_prt = 0.25
    Kasyn_lyso = 7.5e-5 * sim_msec
    Krasyn_lb = 7.5e-5 * sim_msec
    Kasyn_lb = 5 * sim_mM

    # LDOPA uptake
    Vaadc_max = 2.78e-6
    Kaadc = 0.13
    Vtran_max = 5.11e-7
    sTYR = 63e-3
    sTRP = 82e-3
    Ksld = 32e-3
    Kstyr = 64e-3
    Kstrp = 15e-3
    sLD = 3.63685e-3

    # Stimulation
    Istim = Istim
    delay = int(500 / dt) #500
    duration = int(2000 / dt) #200

    delay_steps = int(delay_ms / dt)
    stim_duration_steps = int(stim_duration_ms / dt)

    pulse_width_steps = int(pulse_width_ms / dt)
    pulse_period_ms = 1000 / stim_freq_hz
    pulse_period_steps = int(pulse_period_ms / dt)

    # Convert pA into the scale used by your model
    # Your code suggests 300 pA was represented as 300 * 1e-6
    Istim = stim_amp_pA * 1e-6 #1e-6

    sigg1 = 0
    sigg2 = 0
    phier = 0
    phimt = 0
    ada = 1
    counttt = 1
    lam = 0.0001

    enfatra = gl
    enfamit = mt

    # Main simulation loop
    for k in range(Ttime):

        if k > int(5000 / dt):
            enfatra = gl
            enfamit = mt
        else:
            enfatra = 1
            enfamit = 1
        if Ca_mt > 0.019:
            sigg1 = 1
        if phier > 0.0:
            sigg2 = 1
        if sigg1 == 1:
            Sig_mts = 0.01
        if sigg2 == 1:
            Sig_ers = 0.01

        k_1pc = 1.0 / (1.0 + 0.1 / ATP)
        k_1nk = 0.37 / (1.0 + 0.094 / ATP)

        # Soma - Membrane potential
        VD = V_snc / V_tau

        stim_start = delay_steps
        stim_end = delay_steps + stim_duration_steps

        # HCN current
        kf_free = 0.006 / (1.0 + np.exp((V_snc + 87.7) / 6.45))
        kf_bnd = 0.0268 / (1.0 + np.exp((V_snc + 94.2) / 13.3))
        kf_hcn = kf_free * P_c + kf_bnd * (1.0 - P_c)
        kr_free = 0.08 / (1.0 + np.exp(-(V_snc + 51.7) / 7.0))
        kr_bnd = 0.08 / (1.0 + np.exp(-(V_snc + 35.5) / 7.0))
        kr_hcn = kr_free * P_o + kr_bnd * (1.0 - P_o)

        # Calcium binding proteins
        CaCalb = Calbtot - Calb
        J_calb = kcal_1 * Calb * Ca_i - kcal_2 * CaCalb

        CaCam = Camtot - Cam
        kcam_cb = 12000.0 * (Ca_i ** 2.0)
        kcam_nb = 3.7e6 * (Ca_i ** 2.0)
        alpha_cam = kcam_cb * kcam_nb * (1.0 / (kcam_cb + kcam_nd) + 1.0 / (kcam_cd + kcam_nd))
        beta_cam = kcam_cd * kcam_nd * (1.0 / (kcam_cb + kcam_nd) + 1.0 / (kcam_cd + kcam_nd))
        J_cam = alpha_cam * Cam - beta_cam * CaCam

       # Ca2+ pump activation
        K_pci = (173.6 / (1.0 + CaCam / 5e-5) + 6.4) * 1e-5
        P_E1Spc = 1.0 / (1.0 + K_pci / Ca_i)
        P_E1pc = 1.0 - P_E1Spc
        alpha_pc = k_1pc * P_E1Spc + k_3pc * P_E1pc

        V_Ca = 0.5 * np.log(Ca_o / Ca_i)
        h_cal = 0.00045 / (0.00045 + Ca_i)
        I_CaL = (g_cal * m_cal * h_cal * ((Ca_i * Ca_o) ** 0.5) * np.sinh(VD - V_Ca)) / (np.sinh(VD) / VD)

        K_pmca = k_pmca * ((10.56 * CaCam) / (CaCam + 5e-5) + 1.2)
        I_pmca = K_pmca * (k_1pc * P_E1Spc * y_pc - k_2pc * P_E2Spc * (1.0 - y_pc)) * 1.0
        Dr = (1.0 + 0.001 * ((Na_i ** 3.0) * Ca_o + (Na_o ** 3.0) * Ca_i)) * (1.0 + Ca_i / 0.0069)
        I_xm = (k_xm * ((Na_i ** 3.0) * Ca_o * np.exp(dell * VD) - (Na_o ** 3.0) * Ca_i * np.exp((dell - 1.0) * VD))) / Dr
        J_ca = (-1.0 / (2.0 * F * vol_cyt)) * ((I_CaL  + 2.0 * I_pmca)- 2.0 * I_xm)

        V_Na = np.log(Na_o / Na_i)
        O_na = (m_na ** 3.0) * h_na
        I_Na = (g_na * O_na * ((Na_i * Na_o) ** 0.5) * np.sinh(0.5 * (VD - V_Na))) / (np.sinh(0.5 * VD) / (0.5 * VD))
        I_Nalk = (g_nalk * ((Na_i * Na_o) ** 0.5) * np.sinh(0.5 * (VD - V_Na))) / (np.sinh(0.5 * VD) / (0.5 * VD))
        I_NaHCN = (g_nahcn * O_hcn * ((Na_i * Na_o) ** 0.5) * np.sinh(0.5 * (VD - V_Na))) / (np.sinh(0.5 * VD) / (0.5 * VD))
        P_E1Snk = 1.0 / (1.0 + (K_nknai / Na_i) * (1.0 + K_i / K_nkki))
        Na_eff = Na_o * np.exp(-0.82 * VD)
        P_E2Snk = 1.0 / (1.0 + (K_nknao / Na_eff) * (1.0 + K_o / K_nkko))
        I_nk = k_nk * (k_1nk * P_E1Snk * y_nk - k_2nk * P_E2Snk * (1.0 - y_nk)) * 1.0
        J_Na = (-1.0 / (F * vol_cyt)) * (3.0 * I_nk + 3.0 * I_xm + I_Na + I_Nalk + I_NaHCN)

        P_E1Dnk = 1.0 / (1.0 + (K_nkki / K_i) * (1.0 + Na_i / K_nknai))
        alpha_nk = k_1nk * P_E1Snk + k_3nk * P_E1Dnk
        P_E2Dnk = 1.0 / (1.0 + (K_nkko / K_o) * (1.0 + Na_eff / K_nknao))
        beta_nk = k_2nk * P_E2Snk + k_4nk * P_E2Dnk

        V_K = np.log(K_o / K_i)
        O_sk = (Ca_i ** 4.2) / (0.00035 ** 4.2 + Ca_i ** 4.2)
        I_Ksk = (g_ksk * O_sk * ((K_i * K_o) ** 0.5) * np.sinh(0.5 * (VD - V_K))) / (np.sinh(0.5 * VD) / (0.5 * VD))
        O_kdr = m_kdr ** 3.0
        I_Kdr = g_kdr * O_kdr * (V_snc - V_K * V_tau)
        O_kir = 1.0 / (1.0 + np.exp((V_snc + 85.0) / 12.1))
        I_Kir = g_kir * O_kir * (V_snc - V_K * V_tau)
        I_K = I_Ksk + I_Kdr + I_Kir
        J_K = (-1.0 / (F * vol_cyt)) * (I_K - 2.0 * I_nk)

        if k == 100:
             print(f"J_Na={J_Na:.6e}, J_K={J_K:.6e}, J_Ca={J_Ca:.6e}")
             print(f"Iext at 300pA would be = {300 * 1e-6:.6e}")

        # ER
        J_pump = k_pump * Ca_i * ATP
        J_ch = k_ch * ((Ca_i ** 2.0) / (K1 ** 2.0 + Ca_i ** 2.0)) * (Ca_er - Ca_i)
        J_leak = k_leak * (Ca_er - Ca_i)

        # Mito
        J_out = (k_out * ((Ca_i ** 2.0) / (K3 ** 2.0 + Ca_i ** 2.0)) + k_m) * Ca_mt
        J_in = k_in * ((Ca_i ** 8.0) / (K2 ** 8.0 + Ca_i ** 8.0))

        # Calcium dynamics
        J_Ca = J_ca - 1 * (J_calb + 4.0 * J_cam) - J_pump + J_ch + J_leak - J_in + J_out

        adca = 0

        # Terminal
        Vsynt = Vsynt_max / (((Ksynt / (adca + Ca_i)) ** 4) + 1)
        jsynt = (Vsynt / (1 + ((Ktyr / TYR) * (1 + (cda / Kicda) + (eda / Kieda)))))

        jvmat = ada * Vcda_max * MM_kin(cda, Kcda, 1)

        jida = kmao * cda

        # ATP-dependent DA packing
        ada = 0.001 * (np.exp(3 * ATP))

        # ATP-dependent vesicle recycling
        nRRP_val = 1 * (np.exp(0.7 * ATP))

        prob = 0.14 * MM_kin((adca + Ca_i), krel, 4)
        jrel = psi * nRRP_val * prob

        jdat = Veda_max * MM_kin(eda, Keda, 1)

        jeda = kcomt * eda

        jldopa = Vaadc_max * MM_kin(LDOPA, Kaadc, 1)

        # Energy metabolism
        V_pumps1 = 1 * (1.0 / (F * vol_cyt)) * (I_nk + I_pmca)
        V_pumps2 = 1 * (jvmat)
        V_pumps3 = 100 * jrel
        v_stim = 0
        v_stim1 = 0.0 * (I_nk + I_pmca)
        J_er_val = (beta_er / rho_er) * (J_pump)

        V_id[k] = V_pumps1
        V_dp[k] = V_pumps2
        V_er[k] = J_er_val
        V_rel[k] = V_pumps3

        V_pumps = V_id[k] + V_dp[k] + V_er[k] + V_rel[k]

        eps = 1e-12

        if not np.isfinite(ATP) or ATP <= 0:
            raise ValueError(
                f"Invalid ATP before uADP at k={k}, time={k*dt:.4f} ms, "
                f"ATP={ATP}, stim_amp_pA={stim_amp_pA}, "
                f"V_snc={V_snc}, Ca_i={Ca_i}, Na_i={Na_i}, K_i={K_i}"
            )

        uADP = Q_adk ** 2.0 + 4.0 * Q_adk * (ANP / ATP - 1.0)
        # np.isnan(uADP)

        if not np.isfinite(uADP) or uADP < 0:
            raise ValueError(
                f"Invalid uADP at k={k}, time={k*dt:.4f} ms, "
                f"uADP={uADP}, ATP={ATP}, ANP={ANP}, Q_adk={Q_adk}, "
                f"AMP_if_ADP_zero={ANP-ATP}, stim_amp_pA={stim_amp_pA}, "
                f"V_snc={V_snc}, Ca_i={Ca_i}, Na_i={Na_i}, K_i={K_i}"
            )

        ADP = (ATP / 2.0) * (-Q_adk + uADP ** 0.5)
        Cr = PCr_tot - PCr
        V_ck = 0 * (kf_ck * PCr * ADP - kr_ck * Cr * ATP)

        ATP_inh = ((1.0 + nATP * (ATP / KI_ATP)) / (1.0 + ATP / KI_ATP)) ** 4.0
        V_pk = Vmax_pk * (GAP / (GAP + Km_GAP_pk)) * (ADP / (ADP + Km_ADP_pk)) * ATP_inh
        pa = 1
        V_op = enfamit * Vmax_op * ((pa * PYR) / ((pa * PYR) + Km_PYR_op)) * (ADP / (ADP + Km_ADP_op)) * (1.0 / (1.0 + 0.1 * (ATP / ADP)))

        AMP = ANP - (ATP + ADP)
        AMP_act = ((1.0 + AMP / Ka_AMP) / (1.0 + nAMP * (AMP / Ka_AMP))) ** 4.0
        V_pfk = Vmax_pfk * (F6P / (F6P + Km_F6P_pfk)) * (ATP / (ATP + Km_ATP_pfk)) * (F26P / (F26P + Km_F26P_pfk)) * ATP_inh * AMP_act

        eta_ldh = 1 - beta_ldh_ros * ((ROS ** 4) / ((ROS ** 4) + (Kldh_ros ** 4)))
        V_ldh = 1 * eta_ldh * (kf_ldh * PYR - kr_ldh * LAC)
        V_lac = Vlac_0 * (1.0 + v_stim1 * K_lac) - K_lac_eff * LAC

        V_hk = enfatra * Vmax_hk * (ATP / (ATP + Km_ATP_hk)) * ((1.0 + (F6P / KI_F6P) ** 4.0) ** -1.0) * GLCe
        AMP_pfk2 = ((AMP / Kamp_pfk2) ** nh_amp) / (1.0 + (AMP / Kamp_pfk2) ** nh_amp)
        V_pfk2 = Vmaxf_pfk2 * (ATP / (ATP + Km_ATP_pfk2)) * (F6P / (F6P + Km_F6P_pfk2)) * AMP_pfk2 - Vmaxr_pfk2 * (F26P / (F26P + Km_F26P_pfk2))

        V_ATPase = Vmax_ATPase * (ATP / (ATP + Km_ATP)) * (1.0 + v_stim)
        dAMP_dATP = -1.0 + Q_adk / 2.0 - (0.5 * (uADP ** 0.5)) + Q_adk * (ANP / (ATP * (uADP ** 0.5)))

        GSSG = (GSH_tot - GSH) / 2
        NADP = NADPH_tot - NADPH
        # Prevent division by zero
        if NADP < 1e-12:
            NADP = 1e-12
        Vppp = Vmax_ppp * (F6P / (F6P + Km_F6P_pfk)) / (1 + ((NADPH / NADP) / Ki_nadph))
        Vgr = kf_gr * GSSG * NADPH - kr_gr * GSH * NADP

        # PD pathology pathways
        eta_op = eta_op_max - beta_op_asyn * (((ASYNA ** 4) / ((ASYNA ** 4) + (Kasyn ** 4))))

        Vros_leak = (0.5282 / ATP) * (1 - eta_op) * V_op

        Vros_cat = Kros_cat * ROS

        Vros_dopa = 0 * Kros_dopa * (((ASYNA ** 4) / ((ASYNA ** 4) + (Kasyn ** 4))))

        Vros_dox = Kros_dox * GSH * ROS

        Vasyn_syn = Kasyn_syn

        Vasyn_ox = Kasyn_ox * ROS * ASYN

        Vasyn_to = Kasyn_to * ASYN

        Vasyn_agg = Krasyn_agg * ASYNA * (((ASYNA ** 6) / ((ASYNA ** 6) + (Kasyn_agg ** 6))))

        Ub = Ub_tot - ASYNT
        Vasyn_tag = Kasyn_tag * ASYNA * Ub * ATP

        Vasyn_prt = Krasyn_prt * ASYNT * ATP * (1 - beta_asyn_prt * (((ASYNG ** 4) / ((ASYNG ** 4) + (Kasyn_prt ** 4)))))

        Vasyn_lyso = Kasyn_lyso * ASYNG * ATP

        Vasyn_lb = Krasyn_lb * ASYNG * (((ASYNG ** 6) / ((ASYNG ** 6) + (Kasyn_lb ** 6))))

        V_pro[k] = 25 * Vasyn_prt + 1 * (3 * Vasyn_tag + 10 * Vasyn_lyso)

        # Stimulation
        # Ibg = 0
        # if k < delay + duration and k > delay:
        #      Iapp = Ibg + Istim
        # else:
        #      Iapp = Ibg

        # Stimulation
        Ibg = 0

        if stim_start <= k < stim_end:
          time_since_stim_start = k - stim_start

          # Pulse is ON during the first 10 ms of each 50 ms cycle
          if (time_since_stim_start % pulse_period_steps) < pulse_width_steps:
            Iapp = Ibg + Istim
          else:
            Iapp = Ibg
        else:
          Iapp = Ibg

        Iext = Iapp
        # print(Iext)
        #Iext = Istim

        if k > int(3000 / dt) and k < int(7500 / dt):
             sLD = 3.63685e-3
        else:
             sLD = 3.63685e-3

        # Differential equations
        V_sncnxt = V_snc + (((F * vol_cyt) / (C_sp * A_pmu)) * (J_Na + J_K + 2.0 * J_Ca + Iext)) * dt
        Ca_inxt = Ca_i + (J_Ca) * dt
        Na_inxt = Na_i + (J_Na) * dt
        K_inxt = K_i + (J_K) * dt
        Calbnxt = Calb + (-J_calb) * dt
        Camnxt = Cam + (-J_cam) * dt
        m_calnxt = m_cal + ((1.0 / (1.0 + np.exp(-(V_snc + 15.0) / 7.0)) - m_cal) / (7.68 * np.exp(-(((V_snc + 65.0) / 17.33) ** 2.0)) + 0.7231)) * dt
        m_nanxt = m_na + (A_mna * np.exp(za_mna * VD) * (1.0 - m_na) - B_mna * np.exp(-zb_mna * VD) * m_na) * dt
        h_nanxt = h_na + (A_hna * np.exp(za_hna * VD) * (1.0 - h_na) - B_hna * np.exp(-zb_hna * VD) * h_na) * dt
        O_hcnnxt = O_hcn + (kf_hcn * (1.0 - O_hcn) - kr_hcn * O_hcn) * dt
        m_kdrnxt = m_kdr + ((1.0 / (1.0 + np.exp(-(V_snc + 25.0) / 12.0)) - m_kdr) / (18.0 / (1.0 + np.exp((V_snc + 39.0) / 8.0)) + 1.0)) * dt
        y_pcnxt = y_pc + (beta_pc * (1.0 - y_pc) - alpha_pc * y_pc) * dt
        y_nknxt = y_nk + (beta_nk * (1.0 - y_nk) - alpha_nk * y_nk) * dt
        ATPusednxt = ATPused + (-ATPused + (1.0 / (F * vol_cyt)) * (I_nk + I_pmca)) * dt
        Ca_ernxt = Ca_er + ((beta_er / rho_er) * (J_pump - (J_ch + J_leak))) * dt
        Ca_mtnxt = Ca_mt + ((beta_mt / rho_mt) * (J_in - J_out)) * dt
        cdanxt = cda + (jsynt + jdat - jvmat - jida + jldopa) * dt
        vdanxt = vda + (jvmat - jrel) * dt
        edanxt = eda + (jrel - jdat - jeda) * dt
        calnxt = cal + (-k3f * (Sig_ers * cal) + k3b * (cai_cal)) * dt
        cai_calnxt = cai_cal + (k3f * (Sig_ers * cal) - k3b * (cai_cal) - k4f * (cai_cal)) * dt
        cal_actnxt = cal_act + (k4f * (cai_cal) - k5f * (cal_act * casp12) + k5b * (cal_act_casp12)) * dt
        casp12nxt = casp12 + (-k5f * (cal_act * casp12) + k5b * (cal_act_casp12)) * dt
        cal_act_casp12nxt = cal_act_casp12 + (k5f * (cal_act * casp12) - k5b * (cal_act_casp12) - k6f * (cal_act_casp12)) * dt
        casp12_actnxt = casp12_act + (k6f * (cal_act_casp12) - k7f * (casp12_act * casp9) + k7b * (casp12_act_casp9)) * dt
        casp9nxt = casp9 + (-k7f * (casp12_act * casp9) + k7b * (casp12_act_casp9)) * dt
        casp12_act_casp9nxt = casp12_act_casp9 + (k7f * (casp12_act * casp9) - k7b * (casp12_act_casp9) - k8f * (casp12_act_casp9)) * dt
        casp9_actnxt = casp9_act + (k8f * (1 * casp12_act_casp9) + k9b * (casp9_act_casp3) - k9f * (casp9_act * casp3) + 1 * k28f * Cytc_casp9 - k12f * casp9_act * IAP + k12b * casp9_act_IAP) * dt
        casp3nxt = casp3 + (-k9f * (casp9_act * casp3) + k9b * (casp9_act_casp3)) * dt
        casp9_act_casp3nxt = casp9_act_casp3 + (-k10f * (casp9_act_casp3) - k9b * (casp9_act_casp3) + k9f * (casp9_act * casp3)) * dt
        casp3_actnxt = casp3_act + (k10f * (casp9_act_casp3) - k11f * (casp9_act * casp3_act) - k13f * casp3_act * IAP + k13b * casp3_act_IAP) * dt
        apopnxt = apop + (k11f * (casp9_act * casp3_act)) * dt

        F6Pnxt = F6P + (V_hk - (V_pfk - V_pfk2) - Vppp * (1 / 6)) * dt
        F26Pnxt = F26P + (V_pfk2) * dt
        GAPnxt = GAP + (2.0 * V_pfk - V_pk) * dt
        PYRnxt = PYR + (V_pk - (V_op + V_ldh)) * dt
        LACnxt = LAC + (2.25 * V_ldh + V_lac) * dt
        ATPnxt = ATP + (((1 * (1 * 2.0 * V_pk + 15.0 * eta_op * V_op + V_ck)) - (V_hk + V_pfk + V_pfk2 + V_ATPase + V_pumps + 25 * Vasyn_prt + 1 * (3 * Vasyn_tag + 10 * Vasyn_lyso))) * ((1.0 - dAMP_dATP) ** -1.0)) * dt
        PCrnxt = PCr + (-V_ck) * dt
        ROSnxt = ROS + (Vros_leak + Vros_ex - Vros_cat + Vros_dopa - Vros_dox) * dt
        ASYNnxt = ASYN + (Vasyn_syn - Vasyn_ox - Vasyn_to) * dt
        ASYNAnxt = ASYNA + (Vasyn_ox - Vasyn_agg - Vasyn_tag) * dt
        ASYNTnxt = ASYNT + (Vasyn_tag - Vasyn_prt) * dt
        ASYNGnxt = ASYNG + (Vasyn_agg - Vasyn_lyso - Vasyn_lb) * dt
        LBnxt = LB + (Vasyn_lb) * dt

        ROS_mitnxt = ROS_mit + (k29f * Sig_mts * Mit) * dt
        PTP_mit_actnxt = PTP_mit_act + (k30f * ROS_mit * PTP_mit) * dt
        Cytc_mitnxt = Cytc_mit + (-k31f * PTP_mit_act * Cytc_mit) * dt
        Cytcnxt = Cytc + (-k27f * Cytc * casp9 + k27b * Cytc_casp9 + k31f * PTP_mit_act * Cytc_mit) * dt
        Cytc_casp9nxt = Cytc_casp9 + (k27f * Cytc * casp9 - k27b * Cytc_casp9 - k28f * Cytc_casp9) * dt
        IAPnxt = IAP + (-k12f * casp9_act * IAP + k12b * casp9_act_IAP - k13f * casp3_act * IAP + k13b * casp3_act_IAP) * dt
        casp9_act_IAPnxt = casp9_act_IAP + (k12f * casp9_act * IAP - k12b * casp9_act_IAP) * dt
        casp3_act_IAPnxt = casp3_act_IAP + (k13f * casp3_act * IAP - k13b * casp3_act_IAP) * dt

        NADPHnxt = NADPH + (2 * Vppp - Vgr) * dt
        GSHnxt = GSH + (2 * Vgr - 2 * Vros_dox) * dt

        LDOPAnxt = LDOPA + (((Vtran_max * sLD) / (Ksld * (1 + (sTYR / Kstyr) + (sTRP / Kstrp)) + sLD)) - jldopa) * dt

        # Update state variables
        V_snc = V_sncnxt
        m_cal = m_calnxt
        m_kdr = m_kdrnxt
        m_na = m_nanxt
        h_na = h_nanxt
        O_hcn = O_hcnnxt
        Calb = Calbnxt
        Cam = Camnxt
        y_nk = y_nknxt
        y_pc = y_pcnxt
        K_i = K_inxt
        Na_i = Na_inxt
        Ca_i = Ca_inxt
        Ca_er = Ca_ernxt
        Ca_mt = Ca_mtnxt
        cda = cdanxt
        vda = vdanxt
        eda = edanxt
        ATPused = ATPusednxt
        cal = calnxt
        cai_cal = cai_calnxt
        cal_act = cal_actnxt
        casp12 = casp12nxt
        cal_act_casp12 = cal_act_casp12nxt
        casp12_act = casp12_actnxt
        casp9 = casp9nxt
        casp12_act_casp9 = casp12_act_casp9nxt
        casp9_act = casp9_actnxt
        casp3 = casp3nxt
        casp9_act_casp3 = casp9_act_casp3nxt
        casp3_act = casp3_actnxt
        apop = apopnxt
        ROS_mit = ROS_mitnxt
        PTP_mit_act = PTP_mit_actnxt
        Cytc_mit = Cytc_mitnxt
        Cytc = Cytcnxt
        Cytc_casp9 = Cytc_casp9nxt
        IAP = IAPnxt
        casp9_act_IAP = casp9_act_IAPnxt
        casp3_act_IAP = casp3_act_IAPnxt
        NADPH = NADPHnxt
        GSH = GSHnxt
        F6P = F6Pnxt
        F26P = F26Pnxt
        GAP = GAPnxt
        PYR = PYRnxt
        LAC = LACnxt
        ATP = ATPnxt
        PCr = PCrnxt
        ROS = ROSnxt
        ASYN = ASYNnxt
        ASYNA = ASYNAnxt
        ASYNT = ASYNTnxt
        ASYNG = ASYNGnxt
        LB = LBnxt
        LDOPA = LDOPAnxt

        # Store values
        V_snc_array[k] = V_snc

        # Detect spikes - detect when voltage crosses threshold upward
        if k > 0:
            if V_snc >= -20 and V_snc_array[k-1] < -20: # and V_snc <= 80:
                snc_firings.append([k, 1])

        nai_array[k] = Na_i
        ki_array[k] = K_i
        cai_array[k] = Ca_i
        atpused_array[k] = ATPused
        apop_array[k] = apop
        eda_array[k] = eda
        cda_array[k] = cda
        vda_array[k] = vda
        ros_mit_array[k] = ROS_mit
        calb_array[k] = Calb
        cam_array[k] = Cam
        caer_array[k] = Ca_er
        camt_array[k] = Ca_mt
        ATP_array[k] = ATP
        LAC_array[k] = LAC
        PYR_array[k] = PYR
        GAP_array[k] = GAP
        GSH_array[k] = GSH
        F6P_array[k] = F6P
        F26P_array[k] = F26P
        PCr_array[k] = PCr
        NADPH_array[k] = NADPH
        ROS_array[k] = ROS
        ASYN_array[k] = ASYN
        ASYNA_array[k] = ASYNA
        ASYNT_array[k] = ASYNT
        ASYNG_array[k] = ASYNG
        LB_array[k] = LB
        LDOPA_array[k] = LDOPA
        Iext_array[k] = Iext
        ADP_array[k] = ADP
        uADP_array[k] = uADP
        

        phi_er[k]=phier
        phi_mt[k]=phimt
        # print(k*dt)
        if k % 1000 == 0:
            print(f"Time: {k * dt:.1f} ms")

    # Calculate phi_er and phi_mt
    phi_er = np.log(cai_array / caer_array)
    phi_mt = np.log(cai_array / camt_array)

    # Convert spike timings
    snc_firings1 = ConvertAPtoST(snc_firings, 1)

    # Calculate firing frequency
    base1 = 1 / 2

    if len(snc_firings1) > 0:
        sncfrequency = len(snc_firings1) / (2 * base1 * dt * Ttime * 1e-3) #
    else:
        sncfrequency = 0

    if len(snc_firings1) > 0:

        sncfrequency_stim = len(snc_firings1) / (2 * base1 * dt * Ttime * 1e-3) #
    else:
        sncfrequency = 0

    # Plotting
    sec = 0.001

    # Figure 1
    fig1 = plt.figure(figsize=(19.2, 9.55))
    sizz = 15

    plt.subplot(3, 1, 1)
    if len(snc_firings1) > 0:
        plt.plot(sec * dt * snc_firings1[:, 0], snc_firings1[:, 1], 'k.', markersize=10)
    plt.ylabel('# of neurons', fontweight='bold')
    plt.title(f'SNc firings (Freq = {sncfrequency:.2f} Hz)', fontsize=sizz, fontweight='bold')
    plt.xlim([0, sec * dt * Ttime])

    plt.subplot(3, 1, 2)
    plt.plot(sec * dt * np.arange(len(V_snc_array)), V_snc_array, 'b')
    plt.ylabel('V (mV)', fontweight='bold')
    plt.title('Membrane potential', fontsize=sizz, fontweight='bold')

    plt.subplot(3, 1, 3)
    plt.plot(sec * dt * np.arange(len(cai_array)), cai_array, 'r')
    plt.xlabel('Time (sec)', fontsize=sizz, fontweight='bold')
    plt.ylabel('Ca²⁺ conc. (mM)', fontweight='bold')
    plt.title('Ca²⁺ conc.', fontsize=sizz, fontweight='bold')

    plt.tight_layout()

    # Figure 2
    fig2 = plt.figure(figsize=(19.2, 9.55))

    plt.subplot(4, 1, 1)
    plt.plot(sec * dt * np.arange(len(ros_mit_array)), ros_mit_array, 'r')
    plt.ylabel('ROS_mit', fontweight='bold')
    plt.title('ROS_mit', fontsize=sizz, fontweight='bold')

    plt.subplot(4, 1, 2)
    plt.plot(sec * dt * np.arange(len(apop_array)), apop_array, 'r')
    plt.ylabel('Apoptosis signal', fontweight='bold')
    plt.title('Apoptosis signal', fontsize=sizz, fontweight='bold')


    plt.subplot(4, 1, 1)
    # plt.subplot(4, 1, 3)
    plt.plot(sec * dt * np.arange(len(atpused_array)), atpused_array, 'k')
    plt.ylabel('ATPused (mM)', fontweight='bold')
    plt.title('ATPused', fontsize=sizz, fontweight='bold')

    plt.subplot(4, 1, 4)
    plt.plot(sec * dt * np.arange(len(eda_array)), eda_array, 'b')
    plt.xlabel('Time (sec)', fontsize=sizz, fontweight='bold')
    plt.ylabel('eDA conc. (mM)', fontweight='bold')
    plt.title('eDA conc.', fontsize=sizz, fontweight='bold')

    plt.tight_layout()

    # Figure 3
    fig3 = plt.figure(figsize=(19.2, 9.55))

    plt.subplot(4, 1, 1)
    plt.plot(sec * dt * np.arange(len(calb_array)), calb_array, 'r')
    plt.ylabel('Calb conc. (mM)', fontweight='bold')
    plt.title('Calb', fontsize=sizz, fontweight='bold')

    plt.subplot(4, 1, 2)
    plt.plot(sec * dt * np.arange(len(cai_array)), cai_array, 'k')
    plt.ylabel('Ca_i conc. (mM)', fontweight='bold')
    plt.title('Ca_i', fontsize=sizz, fontweight='bold')

    plt.subplot(4, 1, 3)
    plt.plot(sec * dt * np.arange(len(caer_array)), caer_array, 'k')
    plt.ylabel('ER Ca²⁺ conc. (mM)', fontweight='bold')
    plt.title('ER Calcium', fontsize=sizz, fontweight='bold')
    plt.axhline(y=np.mean(caer_array), color='r', linestyle='--',
                label=f'{np.mean(caer_array):.6f}')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(sec * dt * np.arange(len(camt_array)), camt_array, 'k')
    plt.xlabel('Time (sec)', fontsize=sizz, fontweight='bold')
    plt.ylabel('MT Ca²⁺ conc. (mM)', fontweight='bold')
    plt.title('MT Calcium', fontsize=sizz, fontweight='bold')
    plt.axhline(y=np.mean(camt_array), color='r', linestyle='--',
                label=f'{np.mean(camt_array):.6f}')
    plt.legend()

    plt.tight_layout()

    # Figure 4
    fig4 = plt.figure(figsize=(19.2, 9.55))
    sizz = 10

    plt.subplot(3, 1, 1)
    plt.plot(sec * dt * np.arange(len(F6P_array)), F6P_array, 'r')
    plt.ylabel('F6P conc. (mM)', fontweight='bold')
    plt.title('F6P conc.', fontsize=sizz, fontweight='bold')

    plt.subplot(3, 1, 2)
    plt.plot(sec * dt * np.arange(len(F26P_array)), F26P_array, 'r')
    plt.ylabel('F26P conc. (mM)', fontweight='bold')
    plt.title('F26P conc.', fontsize=sizz, fontweight='bold')

    plt.subplot(3, 1, 3)
    plt.plot(sec * dt * np.arange(len(GAP_array)), GAP_array, 'r')
    plt.xlabel('Time (sec)', fontsize=sizz, fontweight='bold')
    plt.ylabel('GAP conc. (mM)', fontweight='bold')
    plt.title('GAP conc.', fontsize=sizz, fontweight='bold')

    plt.tight_layout()

    # Figure 5
    fig5 = plt.figure(figsize=(19.2, 9.55))

    plt.subplot(4, 1, 1)
    plt.plot(sec * dt * np.arange(len(PYR_array)), PYR_array, 'r')
    plt.ylabel('PYR conc. (mM)', fontweight='bold')
    plt.title('PYR', fontsize=sizz, fontweight='bold')

    plt.subplot(4, 1, 2)
    plt.plot(sec * dt * np.arange(len(LAC_array)), LAC_array, 'r')
    plt.ylabel('LAC conc. (mM)', fontweight='bold')
    plt.title('LAC', fontsize=sizz, fontweight='bold')

    plt.subplot(4, 1, 3)
    plt.plot(sec * dt * np.arange(len(ATP_array)), ATP_array, 'r')
    plt.ylabel('ATP conc. (mM)', fontweight='bold')
    plt.title('ATP conc.', fontsize=sizz, fontweight='bold')

    plt.subplot(4, 1, 4)
    plt.plot(sec * dt * np.arange(len(PCr_array)), PCr_array, 'r')
    plt.xlabel('Time (sec)', fontsize=sizz, fontweight='bold')
    plt.ylabel('PCr conc. (mM)', fontweight='bold')
    plt.title('PCr conc.', fontsize=sizz, fontweight='bold')

    plt.tight_layout()

    # Figure 6
    fig6 = plt.figure(figsize=(19.2, 11))

    plt.subplot(6, 1, 1)
    plt.plot(sec * dt * np.arange(len(ROS_array)), ROS_array, 'r')
    plt.ylabel('ROS conc. (mM)', fontweight='bold', fontsize=8)
    plt.title('ROS conc.', fontsize=sizz, fontweight='bold')
    plt.tick_params(labelsize=8)

    plt.subplot(6, 1, 2)
    plt.plot(sec * dt * np.arange(len(ASYN_array)), ASYN_array, 'r')
    plt.ylabel('α-syn conc. (mM)', fontweight='bold', fontsize=8)
    plt.title('α-syn conc.', fontsize=sizz, fontweight='bold')
    plt.tick_params(labelsize=8)

    plt.subplot(6, 1, 3)
    plt.plot(sec * dt * np.arange(len(ASYNA_array)), ASYNA_array, 'r')
    plt.ylabel('α-syn* conc. (mM)', fontweight='bold', fontsize=8)
    plt.title('α-syn* conc.', fontsize=sizz, fontweight='bold')
    plt.tick_params(labelsize=8)

    plt.subplot(6, 1, 4)
    plt.plot(sec * dt * np.arange(len(ASYNT_array)), ASYNT_array, 'r')
    plt.ylabel('α-syn-tag conc. (mM)', fontweight='bold', fontsize=8)
    plt.title('α-syn-tag conc.', fontsize=sizz, fontweight='bold')
    plt.tick_params(labelsize=8)

    plt.subplot(6, 1, 5)
    plt.plot(sec * dt * np.arange(len(ASYNG_array)), ASYNG_array, 'r')
    plt.ylabel('α-syn-agg conc. (mM)', fontweight='bold', fontsize=8)
    plt.title('α-syn-agg conc.', fontsize=sizz, fontweight='bold')
    plt.tick_params(labelsize=8)

    plt.subplot(6, 1, 6)
    plt.plot(sec * dt * np.arange(len(LB_array)), LB_array, 'r')
    plt.xlabel('Time (sec)', fontsize=sizz, fontweight='bold')
    plt.ylabel('LB conc. (mM)', fontweight='bold', fontsize=8)
    plt.title('LB conc.', fontsize=sizz, fontweight='bold')
    plt.tick_params(labelsize=8)

    plt.subplots_adjust(hspace=0.4)

    # Figure 7
    fig7 = plt.figure(figsize=(19.2, 9.55))

    plt.subplot(2, 1, 1)
    plt.plot(sec * dt * np.arange(len(NADPH_array)), NADPH_array, 'r')
    plt.ylabel('NADPH conc. (mM)', fontweight='bold')
    plt.title('NADPH conc.', fontsize=sizz, fontweight='bold')

    plt.subplot(2, 1, 2)
    plt.plot(sec * dt * np.arange(len(GSH_array)), GSH_array, 'r')
    plt.xlabel('Time (sec)', fontsize=sizz, fontweight='bold')
    plt.ylabel('GSH conc. (mM)', fontweight='bold')
    plt.title('GSH conc.', fontsize=sizz, fontweight='bold')

    plt.tight_layout()

    # Figure 8
    fig8 = plt.figure(figsize=(19.2, 9.55))

    plt.subplot(4, 1, 1)
    plt.plot(sec * dt * np.arange(len(cda_array)), cda_array, 'r')
    plt.ylabel('cDA conc. (mM)', fontweight='bold')
    plt.title('cDA conc.', fontsize=sizz, fontweight='bold')
    plt.axhline(y=np.mean(cda_array), color='b', linestyle='--',
                label=f'{np.mean(cda_array):.6f}')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(sec * dt * np.arange(len(vda_array)), vda_array, 'r')
    plt.ylabel('vDA conc. (mM)', fontweight='bold')
    plt.title('vDA conc.', fontsize=sizz, fontweight='bold')
    plt.axhline(y=np.mean(vda_array), color='b', linestyle='--',
                label=f'{np.mean(vda_array):.1f}')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(sec * dt * np.arange(len(eda_array)), eda_array, 'r')
    plt.ylabel('eDA conc. (mM)', fontweight='bold')
    plt.title('eDA conc.', fontsize=sizz, fontweight='bold')
    plt.axhline(y=np.mean(eda_array), color='b', linestyle='--',
                label=f'{np.mean(eda_array):.6e}')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(sec * dt * np.arange(len(LDOPA_array)), LDOPA_array, 'r')
    plt.xlabel('Time (sec)', fontsize=sizz, fontweight='bold')
    plt.ylabel('LDOPA conc. (mM)', fontweight='bold')
    plt.title('LDOPA conc.', fontsize=sizz, fontweight='bold')
    plt.axhline(y=np.mean(LDOPA_array), color='b', linestyle='--',
                label=f'{np.mean(LDOPA_array):.6e}')
    plt.legend()

    plt.tight_layout()

    # Figure 10
    fig10 = plt.figure(figsize=(19.2, 9.55))
    fig10.suptitle('Energy consumption in different cellular processes', fontsize=14, fontweight='bold')

    plt.subplot(4, 1, 1)
    plt.plot(sec * dt * np.arange(len(V_id)), V_id, 'r')
    plt.ylabel('E_id', fontweight='bold')
    plt.title('Ion dynamics.', fontsize=sizz, fontweight='bold')
    plt.axhline(y=np.mean(V_id), color='b', linestyle='--',
                label=f'{np.mean(V_id):.6e}')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(sec * dt * np.arange(len(V_dp)), V_dp, 'r')
    plt.ylabel('E_dp', fontweight='bold')
    plt.title('Dopamine packing', fontsize=sizz, fontweight='bold')
    plt.axhline(y=np.mean(V_dp), color='b', linestyle='--',
                label=f'{np.mean(V_dp):.6e}')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(sec * dt * np.arange(len(V_rel)), V_rel, 'r')
    plt.ylabel('E_vr', fontweight='bold')
    plt.title('Vesicle recycling', fontsize=sizz, fontweight='bold')
    plt.axhline(y=np.mean(V_rel), color='b', linestyle='--',
                label=f'{np.mean(V_rel):.6e}')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(sec * dt * np.arange(len(V_er)), V_er, 'r')
    plt.xlabel('Time (sec)', fontsize=sizz, fontweight='bold')
    plt.ylabel('E_er', fontweight='bold')
    plt.title('ER', fontsize=sizz, fontweight='bold')
    plt.axhline(y=np.mean(V_er), color='b', linestyle='--',
                label=f'{np.mean(V_er):.6e}')
    plt.legend()

    plt.tight_layout()

    plt.show()

    elapsed_time = time.time() - start_time
    print(f"\nSimulation completed in {elapsed_time:.2f} seconds")

    return {
        'V_snc_array': V_snc_array,
        'cai_array': cai_array,
        'ATP_array': ATP_array,
        'eda_array': eda_array,
        'frequency': sncfrequency,
        'snc_firings1': snc_firings1,
        'Iext_array': Iext_array,
        'ADP_array': ADP_array,
        'uADP_array': uADP_array,
        'atpused_array': atpused_array,
        'ATP_array': ATP_array,
        'V_id' : V_id,
        'V_dp' : V_dp,
        'V_er': V_er ,
        'V_rel' :V_rel
    }

dur = 2000 # ms
gl = [0.0,0.2,0.4,0.6,0.8,1.0, 1.2] # 1
mt = [0.001,0.007,0.04,0.1,0.7,1 ] # 1
durr = dur ##/1000
n_trials = 2
Istim=0.0

# results = SNcATPapopNM(
#             dur = durr,
#             Istim = Istim,
#             gl=gl,
#             mt=mt,
#             filename='ATP_'
#         )*1e-6
results = SNcATPapopNM(dur, gl, mt, Istim, filename='ATP_', delay_ms=500,stim_duration_ms=1000,stim_amp_pA=0,pulse_width_ms=10,stim_freq_hz=20)

stim_amp_pA = list(range(50,301,50))
sncfrequency_stim = []
for amp in stim_amp_pA:
  results = SNcATPapopNM(dur, gl, mt, Istim, filename='ATP_', delay_ms=500,stim_duration_ms=1000,stim_amp_pA=amp,pulse_width_ms=10,stim_freq_hz=20)
  freq = results['frequency']
  sncfrequency_stim.append(freq)

dufour_current= [50,100,150,200,250,300]
dufour_frequency = [5,7,10,13,15,18]

plt.plot(stim_amp_pA, sncfrequency_stim, marker='o',label='SNc model')
plt.plot(dufour_current,dufour_frequency,marker='s',label='dufour')
plt.xlabel('Stimulus amplitude (pA)')
plt.ylabel('Frequency')
plt.xlabel('spike')
plt.legend()
plt.show()

dt=0.1
firings=results['snc_firings1']
firings1=firings[(firings[:, 0] > 500/dt) & (firings[:, 0] <= 1500/dt)]
firings1

sncfrequency_stim = len(firings1) / (10000 * 0.1 * 1e-3)
sncfrequency_stim
