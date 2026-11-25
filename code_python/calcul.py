import math, cmath



def calculZos(er,Wsl,lamda0s,h) :

    wl = Wsl / lamda0s   # W / lamda0
    dl = h / lamda0s     # d / lamda0
    wd = Wsl / h        # W / d

    if((wl<0.0015) or (wl>1)):
                print("wl en dehors du domaine valide")
    if (dl < 0.006) or (dl > 0.06):
            print("dl en dehors du domaine valide")
    if((er<3.8) or (er>9.8)):
            print("er en dehors du domaine valide")
    
    

    if wl <= 0.075:     # équation (13)
        Zos = (73.6 - 2.15 * er
               + (638.9 - 31.37 * er) * wl**0.6
               + (36.23 * math.sqrt(er**2 + 41) - 225) * wd / (wd + 0.876 * er - 2)
               + 0.51 * (er + 2.12) * wd * math.log(100 * dl)
               - 0.753 * er * dl / math.sqrt(wl))
    else:               # équation (15)
        exposant = 1.11 + 0.132 * (er - 27.7) / (100 * dl + 5)
        Zos = (120.75 - 3.74 * er
               + 50 * (math.atan(2 * er) - 0.8) * wd**exposant
               * math.log(100 * dl + math.sqrt((100 * dl)**2 + 1))
               + 14.21 * (1 - 0.458 * er) * (100 * dl + 5.1 * math.log(er) - 13.1)
               * (wl + 0.33)**2)
    return Zos
   


def calculLamdas(er, Wsl, lamda0s, h):
    if (er < 3.8) or (er > 9.8):
        print("er en dehors du domaine valide")

    wl = Wsl / lamda0s   # W / lamda0
    dl = h / lamda0s     # d / lamda0
    wd = Wsl / h         # W / d

    if (wl < 0.0015) or (wl > 1):
        print("wl en dehors du domaine valide")
    if (dl < 0.006) or (dl > 0.06):
        print("dl en dehors du domaine valide")

    if wl <= 0.075:     # équation (12)
        rapport = (0.9217 - 0.277 * math.log(er)
                   + 0.0322 * wd * math.sqrt(er / (wd + 0.435))
                   - 0.01 * math.log(dl)
                   * (4.6 - 3.65 / (er**2 * math.sqrt(wl) * (9.06 - 100 * wl))))
    else:               # équation (14)
        rapport = (1.05 - 0.04 * er
                   + 1.411e-2 * (er - 1.421) * math.log(wd - 2.012 * (1 - 0.146 * er))
                   + 0.111 * (1 - 0.366 * er) * math.sqrt(wl)
                   + 0.139 * (1 + 0.52 * er * math.log(14.7 - er)) * dl * math.log(dl))

    lamdas = rapport * lamda0s    # les formules donnent lamda_s / lamda0
    return lamdas
    
def calculn(er, h, lamda0s, lamdas):
    r = lamda0s / lamdas                 # lamda0 / lamda_s
    u = math.sqrt(er - r**2)
    v = math.sqrt(r**2 - 1)
    a = 2 * math.pi * h * u / lamda0s    # 2*pi*h*u / lamda0
    q0 = a + math.atan(u / v)
    n = math.cos(a) - math.sin(a) / math.tan(q0)   # cot(q0) = 1/tan(q0)
    return n

def CalculWms(er, Zom, h):
    A = Zom / 60 * math.sqrt((er + 1) / 2) + (er - 1) / (er + 1) * (0.23 + 0.11 / er)
    B = 377 * math.pi / (2 * Zom * math.sqrt(er))

    wd = 8 * math.exp(A) / (math.exp(2 * A) - 2)    # formule pour W/d < 2
    if wd > 2:                                      # sinon, formule pour W/d > 2
        wd = 2 / math.pi * (B - 1 - math.log(2 * B - 1)
                            + (er - 1) / (2 * er) * (math.log(B - 1) + 0.39 - 0.61 / er))

    Wms = wd * h
    return Wms

def calculEeff(er, h, Wms):
    wh = Wms / h         # W / d
    eeff = (er + 1) / 2 + (er - 1) / 2 / math.sqrt(1 + 12 / wh)    # 12/wh = 12 h / W
    return eeff

def calculN(Gamma, Zom, Zo, thetam):

    G0 = abs(0.5 * math.log(Zom / Zo))      # réflexion sans transformateur, Gamma(0)

    if G0 <= Gamma:
        print("Pas besoin de transformateur : la réflexion est déjà sous Gamma")
        return 0

    Nreel = math.acosh(G0 / Gamma) / math.acosh(1 / math.cos(thetam))
    N = math.ceil(Nreel)                     # on arrondit au nombre entier supérieur
    return N

def chebyshevT(N, x):
    # polynôme de Chebyshev T_N(x), par la récurrence T_n = 2x T_(n-1) - T_(n-2)
    t_prec=1.0
    t =x
    if N == 0:
        return 1.0
    for _ in range(N - 1):
        t_nouveau = 2 * x * t - t_prec
        t_prec=t
        t =t_nouveau
    return t

def calculZn(Zo, Zom, N, thetam):
    
    s = 1 / math.cos(thetam)                   # sec(thetam)

    # Étape 1 : la constante A (= ondulation Gamma_m)
    G0 = 0.5 * math.log(Zom / Zo)
    A = G0 / chebyshevT(N, s)

    # Étape 2 : les réflexions Gamma_0 ... Gamma_N
    # On échantillonne Gamma(theta) = A e^{-jN theta} T_N(s cos theta)
    # et on extrait les coefficients de sum Gamma_n e^{-2jn theta}
    M = 4 * (N + 1)                            # nombre de points de calcul
    Gamma = []
    for n in range(N + 1):
        somme = 0
        for k in range(M):
            th = math.pi * k / M
            cible = A * cmath.exp(-1j * N * th) * chebyshevT(N, s * math.cos(th))
            somme += cible * cmath.exp(2j * n * th)
        Gamma.append((somme / M).real)

    # Étape 3 : les impédances des N sections
    Zn = [Zo]                                       
    for n in range(N):
        Zn.append(Zn[-1] * math.exp(2 * Gamma[n]))  
    return Zn[1:]

fmin=2*pow(10,9)
fmax=7*pow(10,9)
fos=math.sqrt(fmin*fmax)
fom=(fmin+fmax)/2
lamda0s=299792458/fos
lamda0m=299792458/fom
thetam=math.pi/2 *fmin/fom
Zo=50 

###Paramètre à fixer###
er=4.4 #permittivité du substrat
h=0.8*pow(10,-3)  #épaisseur du substrat
Wsl=0.4*pow(10,-3) #largeur initial de la slotline
Gamma=0.01 #réflexion maximum dans le microstrip

###paramètre à calculer###
Zos=calculZos(er,Wsl,lamda0s,h)
Lamdas=calculLamdas(er,Wsl,lamda0s,h)
n=calculn(er,h,lamda0s,Lamdas)
Zom=n*n*Zos
Wms=CalculWms(er,Zom,h)
Eeff=calculEeff(er,h,Wms)
lamdam=lamda0s/math.sqrt(Eeff)
N=calculN(Gamma,Zom,Zo,thetam)
Zn=calculZn(Zo,Zom,N,thetam)

Wn = []    
Ln = []    
for Z in Zn:
    W = CalculWms(er, Z, h)                  
    eeff = calculEeff(er, h, W)              
    L = lamda0m / (4 * math.sqrt(eeff))       
    Wn.append(W)
    Ln.append(L)

for Z, W, L in zip(Zn, Wn, Ln):
    print(f"Z = {Z:5.10f} ohm   W = {W:.10f}   L = {L:.10f} ")