# RNX2VTX_v3_2.py
# rnx2vtx_v3.0

# v1.0 2023/01/29 : GPS , rinux ver 3.02 , DONT estimate receiver and satelite biases
# v1.1 2023/01/31 : Estimate Differential Code Bias, smooth option
# v1.2 2023/02/05 : Import orbit data 1 time (update)
#                 : This update is intended to address broken orbital data.
# v2.0 2023/02/08 : GLONASS,QZSS,Galileo are supported.
# v2.1 2023/02/10 : Can export unbiased TEC data. (no_bias_TEC option)
#      2023/02/23 : Acceleration update for 1st procedure. (13000s for a day -> 8500s for a day)
# v2.2 2023/03/07 : rinux ver 2.10 are supported.
# v3.0 2023/04/19 : *****************   Attention   ******************************
#                   Earlier versions have fatal flaws. Be sure to use this version or later.
#                   Support for GPS, GLONASS, QZSS, and Galileo to find DCB between their respective frequencies.
# v3.2 2023/05/16 : Added restriction on elevation angle.
#                   Corrected record code to be read.
# v3.2.1 2023/05/23:rinux 2.12 supported
################################################################################################################
#
# USAGE
#
# This program uses electronic reference point data published by the Geospatial Information Authority of Japan.
# From this data, the inter-frequency bias and absolute TEC values for each satellite and receiving station are obtained.
#
# Please store the observation data and orbit data downloaded from GSI as follows.
#
# {download folder}\{country code}\{year:4 digit}\{day(Day of Year):3 digit}\{files}
# download folder ... any
# country code ... Based on the domain of each country. (ex: Japan -> jp, Korea -> kr)
# year ... Specify by 4 digits (ex: 2017,2018)
# day ... day of year(Number of days counted from January 1), specify by 3 digits (ex: 1/3 -> 003)
# files ... ALL Orbit and Observation Data (ex: 00010010.01g,00010010.01l,00010010.01q,00010010.01n,00010010.01o)
#
# Each procedure is executed or not depending on the bool value.
#
# raw_TEC_estimate ... Match the TEC value obtained from the pseudo-distance with the TEC value obtained from the
#                      carrier phase.
# bias_estimate ...... Estimating satellite and receiving station bias.
# smooth (optional) .. Complementary VTEC map.
# no_bias_TEC ........ Generate bias-free tomography files. (Contain time(UTC [hour]), STEC(bias-free), Satelite position
#                      (ECEF,XYZ [m]), Receiver position(ECEF,XYZ [m]))
#
# The program outputs three types of files.
#
# 1.raw TEC file ... Contain matching the TEC value obtained from the pseudo-distance with the TEC value obtained
#                    from the carrier phase
#  output folder ->   mdf\{country code}\{year:4 digit}\{day(Day of Year):3 digit}\{files}
#
# 2.VTEC map file ... Includes VTEC map data and bias for each satellite and receiving station.
#  output folder ->   vtecmap\{country code}\{year:4 digit}\{day(Day of Year):3 digit}\{files}
#
# 3.bias free file ... Contains TEC values without bias and ECEF coordinates of the satellite and receiving station
#  output folder ->   unbias\{country code}\{year:4 digit}\{day(Day of Year):3 digit}\{files}
#
##################################################################################################################

# for procedure
import random
import matplotlib.pyplot as plt
import math
import numpy as np
import datetime
import os
import glob
import time
import pymap3d as pm
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import inv
from scipy.sparse.linalg import spsolve
from scipy.optimize import lsq_linear
import warnings

warnings.simplefilter('error')

# for illustration

np.set_printoptions(edgeitems=1440, precision=4, suppress=True)

rf = 1.0/298.257223563
ra = 6378137.0
rb = ra*(1.0-rf)
re = math.sqrt((ra*ra-rb*rb)/(ra*ra))

maxrec = 1500
maxsat = 120  # all
Gmaxsat = 40
Emaxsat = 40
Rmaxsat = 40
Jmaxsat = 10
threshold = 1.5

vel_light = 2.9979*math.pow(10, 8)
K = 80.62
f_l1_GPS = 1575420000
f_l2_GPS = 1227600000
f_lA_GPS = 1575420000
f_l1_Galileo = 1575420000
f_l5_Galileo = 1176450000
f_l1_QZSS = 1575420000
f_lC_QZSS = 1227600000
f_l5_QZSS = 1176450000
f_l1_GLONASS_base = 1602000000
f_l2_GLONASS_base = 1246000000
f_l1_GLONASS_del = 562500
f_l2_GLONASS_del = 437500

H_ipp = 400.0  # [km]
H1_ipp = 300.0
H2_ipp = 550.0
zenith_threshold = 15.0
null_threshold = 2
Valid_Data_Length = 60

year4 = 2016
year2 = year4 % 100
country = "jp"
days = [106]
version = "rnx2vtx_3.2.2"

tec_folder = "E:/tec"
mdf_folder = "E:/mdf"
vtec_folder = "E:/vtecmap"
unbias_folder = "E:/unbias"

raw_TEC_estimate = True
bias_estimate = True
smooth = False
no_bias_TEC = True

glonass_slot = np.zeros(25, dtype=int)


class BLH:
    # b,l...[degree]
    # h...[km]
    b: float = 0.0
    l: float = 0.0
    h: float = 0.0

    def set(self, b, l, h):
        self.b = b
        self.l = l
        self.h = h

    def to_XYZ(self):
        answer = XYZ()
        n = ra/math.sqrt(1.0-re*re*math.sin(math.radians(self.b))
                         * math.sin(math.radians(self.b)))
        answer.x = (n+self.h)*math.cos(math.radians(self.b)) * \
            math.cos(math.radians(self.l))
        answer.y = (n+self.h)*math.cos(math.radians(self.b)) * \
            math.sin(math.radians(self.l))
        answer.z = ((1-re*re)*n+self.h)*math.sin(math.radians(self.b))
        return answer

    def __str__(self):
        return "[ B: "+str(self.b)+" L: "+str(self.l)+" H: "+str(self.h)+" ]"


class XYZ:

    # x,y,z...[km]
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def set(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_BLH(self):
        # print(type(self))
        # print(self.x)
        X = float(self.x)
        Y = float(self.y)
        Z = float(self.z)
        answer = BLH()
        # 1
        # print(type(X))
        # print(type(Y))
        p = math.sqrt(X*X+Y*Y)
        h = ra*ra-rb*rb
        t = math.atan2(Z*ra, p*rb)  # rad
        answer.l = math.degrees(math.atan2(Y, X))  # deg
        # 2
        answer.b = math.degrees(math.atan2(
            (ra*rb*Z+ra*h*math.sin(t)**3), (ra*rb*p-rb*h*math.cos(t)**3)))  # deg
        # 3
        n = ra/math.sqrt(1-re*re*math.sin(math.radians(answer.b))
                         * math.sin(math.radians(answer.b)))
        # 4
        answer.h = p/math.cos(math.radians(answer.b))-n
        return answer

    def __str__(self):
        return "[ X: "+str(self.x)+" Y: "+str(self.y)+" Z: "+str(self.z)+" ]"

    def __sub__(self, other):
        answer = XYZ()
        answer.set(self.x-other.x, self.y-other.y, self.z-other.z)
        return answer

    def L2(self) -> float:
        siz = self.x**2+self.y**2+self.z**2
        return math.sqrt(siz)

# [m]
# returnのfloatはRECで0,SATで1となるよ�?に標準化した距離パラメータ


def specify_H(rec: XYZ, sat: XYZ, H, t=1.0) -> XYZ:
    # 1[m]
    alpha = 1.0
    eps = 0.001
    dt = 0.01
    point = XYZ()
    point.set(rec.x*(1-t)+sat.x*t, rec.y*(1-t)+sat.y*t, rec.z*(1-t)+sat.z*t)
    # print(t, point.to_BLH())
    if abs(point.to_BLH().h-H) < eps:
        if t < 0.0 or t > 1.0:
            return rec
        else:
            return point
    else:
        dpoint = XYZ()
        dpoint.set(rec.x*(1-t-dt)+sat.x*(t+dt), rec.y*(1-t-dt) +
                   sat.y*(t+dt), rec.z*(1-t-dt)+sat.z*(t+dt))
        fbar_i = (dpoint.to_BLH().h-point.to_BLH().h)/dt
        tbar = t-alpha*(point.to_BLH().h-H)/fbar_i
        # print(t, tbar)
        return specify_H(rec, sat, H, tbar)

# rad


def zenith(Rec: XYZ, Ipp: XYZ):
    rec = [Rec.x, Rec.y, Rec.z]
    ipp = [Ipp.x, Ipp.y, Ipp.z]
    r2i = [0.0 for i in range(3)]
    for i in range(3):
        r2i[i] = ipp[i]-rec[i]
    inner = 0.0
    r2is = 0.0
    ipps = 0.0
    for i in range(3):
        inner += r2i[i]*ipp[i]
        r2is += r2i[i]*r2i[i]
        ipps += ipp[i]*ipp[i]
    cosZ = inner/(math.sqrt(r2is*ipps))
    return math.acos(cosZ)


def r_zenith(Rec: XYZ, Sat: XYZ):
    rec = [Rec.x, Rec.y, Rec.z]
    sat = [Sat.x, Sat.y, Sat.z]
    r2s = [0.0 for i in range(3)]
    for i in range(3):
        r2s[i] = sat[i]-rec[i]
    inner = 0.0
    r2ss = 0.0
    recs = 0.0
    for i in range(3):
        inner += rec[i]*r2s[i]
        r2ss += r2s[i]*r2s[i]
        recs += rec[i]*rec[i]
    cosZ = inner/(math.sqrt(r2ss*recs))
    return math.acos(cosZ)


def kepler(dmk, e):
    thres = pow(10, -14)
    niteration = 0
    ek = dmk
    diff = ek+e*math.sin(ek)-dmk  # [rad]
    while abs(diff) > thres:
        diff = ek+e*math.sin(ek)-dmk
        partial = 1-e*math.cos(ek)
        ek = ek-diff/partial
        niteration += 1
        if niteration > 100:
            print("The calculation was terminated because the iteration of Newton's method in the Kep1er function exceeded 100.")
            break
    return ek


def gtxyz(time1, time0, ele):

    # data format of navigation file
    #       0:IODE  1:Crs  2:delta-n  3:m0
    #       4:Cuc   5:e    6:Cus      7:root-a
    #       8:Toe   9:Cic 10:Omega   11:Cis
    #      12:i0   13:Crc 14:omega   15:OmegaDot
    #      16:iDot 17-27: not used
    GM = 3.986005*math.pow(10, 14)  # [m^3/s^2]
    omega_dot_e = 7.292115*math.pow(10, -5)  # [rad/s]
    # (1)
    # print(ele[7])
    a = ele[7]**2  # [m]
    # (2)
    dnzero = math.sqrt(GM*math.pow(a, -3.0))  # [rad/s]
    # (3)
    tk = (time1-time0)*60.0*60.0  # [s]
    # (4)
    dn = dnzero+ele[2]  # [rad/s]
    # (5)
    dmk = ele[3]+dn*tk  # [rad]
    # (6)
    ek = kepler(dmk, ele[5])  # [rad]
    # (7)
    cosvk = (math.cos(ek)-ele[5])/(1.0-ele[5]*math.cos(ek))
    sinvk = math.sqrt(1.0-ele[5]*ele[5])*math.sin(ek)/(1.0-ele[5]*math.cos(ek))
    vk = math.atan2(sinvk, cosvk)  # [rad]
    # (8)
    phik = vk+ele[14]  # [rad]
    # (9)
    delta_uk = ele[6]*math.sin(2.0*phik)+ele[4]*math.cos(2.0*phik)  # [rad]
    uk = phik+delta_uk  # [rad]
    # (10)
    delta_rk = ele[1]*math.sin(2.0*phik)+ele[13]*math.cos(2.0*phik)  # [m]
    rk = a*(1.0-ele[5]*math.cos(ek))+delta_rk
    # (11)
    delta_dik = ele[11]*math.sin(2.0*phik)+ele[9]*math.cos(2.0*phik)
    dik = ele[12]+delta_dik+ele[16]*tk  # [rad]
    # (12)
    xdashk = rk*math.cos(uk)  # [m]
    ydashk = rk*math.sin(uk)  # [m]
    # (13)
    # [rad]    [rad]   [rad/s]    [rad/2]   [s]   [s]   [rad/s]
    omegak = ele[10]+(ele[15]-omega_dot_e)*tk-ele[8]*omega_dot_e
    # (14)
    dx = xdashk*math.cos(omegak)-ydashk*math.cos(dik)*math.sin(omegak)
    dy = xdashk*math.sin(omegak)+ydashk*math.cos(dik)*math.cos(omegak)
    dz = ydashk*math.sin(dik)
    return [dx, dy, dz]

# Estimate from 1 point


def gtxyz_glonass(time1, time0, ele):
    dt = time1-time0
    x_0 = ele[0]
    y_0 = ele[3]
    z_0 = ele[6]
    u = ele[1]
    v = ele[4]
    w = ele[7]
    a_x = ele[2]
    a_y = ele[5]
    a_z = ele[8]
    dx = x_0+u*dt+(a_x*dt*dt)/2.0
    dy = y_0+v*dt+(a_y*dt*dt)/2.0
    dz = z_0+w*dt+(a_z*dt*dt)/2.0
    return [dx*1000.0, dy*1000.0, dz*1000.0]

# Estimation from 2 point(s)


def gtxyz_glonass_2p(ttime, time0, ele0, time1, ele1):
    dt = time1-time0
    alpha = (ttime-time0)/dt

    x_t1 = ele1[0]
    x_t0 = ele0[0]
    u_t0 = ele0[1]
    a_t0 = ele0[2]
    S_x = x_t1-x_t0-u_t0*dt-(dt*dt*a_t0)/2.0
    dx = x_t0+alpha*dt*u_t0+alpha*alpha*dt*dt*a_t0/2.0+S_x*math.pow(alpha, 3)

    y_t1 = ele1[3]
    y_t0 = ele0[3]
    v_t0 = ele0[4]
    b_t0 = ele0[5]
    S_y = y_t1-y_t0-v_t0*dt-(dt*dt*b_t0)/2.0
    dy = y_t0+alpha*dt*v_t0+alpha*alpha*dt*dt*b_t0/2.0+S_y*math.pow(alpha, 3)

    z_t1 = ele1[6]
    z_t0 = ele0[6]
    w_t0 = ele0[7]
    c_t0 = ele0[8]
    S_z = z_t1-z_t0-w_t0*dt-(dt*dt*c_t0)/2.0
    dz = z_t0+alpha*dt*w_t0+alpha*alpha*dt*dt*c_t0/2.0+S_z*math.pow(alpha, 3)

    return [dx*1000.0, dy*1000.0, dz*1000.0]


# idsat[k] ... k th record satelite id
# oel[k] ... k th record
# tiempo[k] ... k th record time from obs_day 00:00:00UTC [hour]
# i1stsat[k] ... list of k th satelite record number
# for RINEX ver 3.02 / 2.12 for QZSS,Galileo
def load_navi_file(file, obs_day):  # for GPS,Galileo,QZSS
    base_dt = obs_day
    oel = np.zeros((maxrec, 28))  # storing navigation data
    # �?ータの衛星番号用. 例えばG01な�?01
    idsat = np.zeros(maxrec, dtype=int)
    # 基準となるobs_day 00:00:00[UTC]からの経過時刻[hour]
    tiempo = np.zeros(maxrec)
    # 衛星番号ごとの�?ータ番号,G01ならi1stsat[1]に格�?
    i1stsat = [[] for i in range(maxsat)]
    # 全ての行�?��?ータをlinesに格�?
    with open(file, "r") as f_n:
        lines = [s.rstrip() for s in f_n.readlines()]

    L = len(lines)
    idx = 0
    krec = 0
    header_flag = True
    # idx行目に注目
    while idx < L:
        # print(idx,":",lines[idx])
        # ま�?ヘッダーなら、次の�?
        if header_flag:
            idx += 1
        # ヘッダーの終わりなら、次の行に行って、header_flagをFalseにする
        if "END OF HEADER" in lines[idx]:
            header_flag = False
            idx += 1
            if not lines[idx]:
                break
            continue
        # ヘッダーではな�?と�?,idxは時刻が書�?てある�?
        if not header_flag:
            # �?ータの時刻を読み取り
            # 0    5    0    5    0
            # G10 2022 03 04 00 00 00
            idsat[krec] = int(lines[idx][1:3])
            iy = int(lines[idx][4:8])
            imonth = int(lines[idx][9:11])
            iday = int(lines[idx][12:14])
            ih = int(lines[idx][15:17])
            imin = int(lines[idx][18:20])
            sec = float(lines[idx][21:23])
            # print(iday,ih,imin,sec)
            dt2 = datetime.datetime(
                year=iy, month=imonth, day=iday, hour=ih, minute=imin, second=int(sec))
            td = dt2-base_dt
            tiempo[krec] = td.total_seconds()/3600.0
            if not lines[idx]:
                break
            # �?ータ�?7行�??読み取る
            for irec in range(7):
                ii = irec*4
                # �?行に�?ータは4種類ずつ書かれて�?�?
                for j in range(4):
                    try:
                        # 0    5    0    5    0    5    0    5    0
                        #     -8.959323167801D-07 7.537830504589D-03 2.330169081688D-06 5.153672527313D+03
                        m_str = lines[idx+irec+1][4+19*j:4+15+19*j].strip()
                        e_str = lines[idx+irec+1][20+19*j:20+3+19*j].strip()
                    except:
                        idsat[krec] = 0
                        break
                    if not m_str == "" and not e_str == "":
                        oel[krec][ii+j] = float(m_str)*math.pow(10, int(e_str))
                    else:
                        oel[krec][ii+j] = 0.0
            # �?ータを読み終わったら8行データを進める
            idx += 8
            if idx < L and not lines[idx]:
                break
            # �?ータの番号を一つすすめる
            krec += 1

    #
    # finding 1st appearance of satellites
    # i1stsat[isat] ... index which No.[isat] satellite's record start
    #

    for k in range(krec):
        i1stsat[idsat[k]].append(k)
    return idsat, tiempo, oel, i1stsat


def load_GLONASS_nav_file(file, obs_day):
    base_dt = obs_day
    header = True
    # tiempo[k] ... time [hour] of k th record
    # oel[k][0:3] ... x coordinate, pos, vec, acc
    # oel[k][3:6] ... y coordinate, pos, vec, acc
    # oel[k][6:9] ... z coordinate, pos, vec, acc
    # idsat[k] ... satelite id of k th record
    tiempo = np.full((maxrec), 0.0, dtype=float)
    oel = np.full((maxrec, 9), 0.0, dtype=float)
    idsat = np.full((maxrec), 0, dtype=int)

    krec = 0
    with open(file, "r") as f:
        while True:
            line = f.readline()
            # print(line)
            if not line:
                break
            if header:
                if "END OF HEADER" in line:
                    header = False
                    # line=f.readline()
                    continue
            elif not header:
                idsat[krec] = int(line[1:3])
                iy = int(line[4:8])
                imonth = int(line[9:11])
                iday = int(line[12:14])
                ih = int(line[15:17])
                imin = int(line[18:20])
                isec = int(line[21:23])
                dt2 = datetime.datetime(
                    year=iy, month=imonth, day=iday, hour=ih, minute=imin, second=isec)
                td = dt2-base_dt
                tiempo[krec] = td.total_seconds()/3600.0
                for irec in range(3):
                    line = f.readline()
                    if not line:
                        break
                    ii = irec*3
                    for j in range(3):
                        m_str = line[4+j*19:19+j*19]
                        e_str = line[20+j*19:23+j*19]
                        if m_str and e_str:
                            oel[krec][ii+j] = float(m_str)*pow(10, int(e_str))
                        else:
                            oel[krec][ii+j] = 0.0
                krec += 1

    i1stsat = [[] for i in range(Rmaxsat)]

    for k in range(krec):
        i1stsat[idsat[k]].append(k)

    return idsat, tiempo, oel, i1stsat

# ***********************************************************
# [yy]o file reader
# sat_dict, sat_list, L_stec, L_stec_bool, P_stec, P_stec_bool, p_o, obs_day
# sat_dict ... index of other data of sat [sat code]
# sat_list ... list of [sat code]
# L_stec ... [i][j] -> epoch j data of sat_dict[sat code](= i)
# L_stec_bool ... [i][j] -> existness epoch j data of sat_dict[sat code](= i)
# P_stec ... [i][j] -> epoch j data of sat_dict[sat code](= i)
# P_stec_bool ... [i][j] -> existness epoch j data of sat_dict[sat code](= i)
# p_o ... ECEF position of receiver
# obs_day ... datetime type


def load_observ_file_v3_02(file):
    # ECEF position of receiver
    p_o = np.zeros(3)
    with open(file, "r") as f:
        gL_l1_num = -1
        gL_l2_num = -1
        gP_l1_num = -1
        gP_l2_num = -1

        rL_l1_num = -1
        rL_l2_num = -1
        rP_l1_num = -1
        rP_l2_num = -1

        eL_l1_num = -1
        eL_l5_num = -1
        eP_l1_num = -1
        eP_l5_num = -1

        jL_l1_num = -1
        jL_l2_num = -1
        jP_l1_num = -1
        jP_l2_num = -1

        while True:
            s_line = f.readline()
            if "APPROX POSITION XYZ" in s_line:
                p_o[0] = float(s_line[1:14])
                p_o[1] = float(s_line[15:28])
                p_o[2] = float(s_line[29:42])
            if "OBS TYPES" in s_line:
                if s_line[4:6].strip() == "":
                    continue
                wave_sum = int(s_line[4:6])
                sat_type = s_line[0:1]
                if sat_type == "G":
                    for i in range(wave_sum):
                        j = i % 13
                        wave = s_line[7+4*j:10+4*j]
                        if wave == "C1C":
                            gP_l1_num = i
                        elif wave == "C2W":
                            gP_l2_num = i
                        elif wave == "L1C":
                            gL_l1_num = i
                        elif wave == "L2W":
                            gL_l2_num = i
                        if j == 12:
                            s_line = f.readline()
                # if sat_type == "R":
                #     for i in range(wave_sum):
                #         j = i % 13
                #         wave = s_line[7+4*j:10+4*j]
                #         if wave == "C1C":
                #             rP_l1_num = i
                #         elif wave == "C2C":
                #             rP_l2_num = i
                #         elif wave == "L1C":
                #             rL_l1_num = i
                #         elif wave == "L2C":
                #             rL_l2_num = i
                #         if j == 12:
                #             s_line = f.readline()
                if sat_type == "E":
                    for i in range(wave_sum):
                        j = i % 13
                        wave = s_line[7+4*j:10+4*j]
                        if wave == "C1X":
                            eP_l1_num = i
                        elif wave == "C5X":
                            eP_l5_num = i
                        elif wave == "L1X":
                            eL_l1_num = i
                        elif wave == "L5X":
                            eL_l5_num = i
                        if j == 12:
                            s_line = f.readline()
                if sat_type == "J":
                    for i in range(wave_sum):
                        j = i % 13
                        wave = s_line[7+4*j:10+4*j]
                        if wave == "C1X":
                            jP_l1_num = i
                        elif wave == "C5X":
                            jP_l2_num = i
                        elif wave == "L1X":
                            jL_l1_num = i
                        elif wave == "L5X":
                            jL_l2_num = i
                        if j == 12:
                            s_line = f.readline()
            if "GLONASS SLOT / FRQ" in s_line:
                r_num = int(s_line[1:3])
                r_line = math.floor((r_num+7)/8)
                for glns_clm in range(r_line):
                    for i_glns in range(8):
                        glns_num = s_line[5+i_glns*7:7+i_glns*7]
                        glns_slot = s_line[8+i_glns*7:10+i_glns*7]
                        if glns_num != "  " and glns_slot != "  ":
                            glonass_slot[int(glns_num)] = int(glns_slot)
                    s_line = f.readline()
            if "END OF HEADER" in s_line:
                break
            if not s_line:
                break
        os.makedirs(
            "{md}/{c}/{y:04d}/{d:03d}".format(md=mdf_folder, c=country, y=year4, d=gday), exist_ok=True)
        glonass_slot_file = "{md}/{c}/{y:04d}/{d:03d}/glonass.txt".format(
            md=mdf_folder, c=country, y=year4, d=gday)
        if os.path.exists(glonass_slot_file):
            pass
        else:
            with open(glonass_slot_file, "w") as gsf:
                for i in range(25):
                    print(i, glonass_slot[i], file=gsf)

        P_stec = np.full((120, 2880), -np.inf, dtype=float)
        P_stec_bool = np.full((120, 2880), False, dtype=bool)
        L_stec = np.full((120, 2880), -np.inf, dtype=float)
        L_stec_bool = np.full((120, 2880), False, dtype=bool)
        sat_dict = {}
        sat_list = []

        while True:
            s_line = f.readline()
            # print(s_line)
            if not s_line:
                break
            iy_str = s_line[2:6]
            if int(iy_str) != year4:
                s_line = f.readline()
            else:
                iyear = int(s_line[2:6])
                imonth = int(s_line[7:9])
                iday = int(s_line[10:12])
                obs_day = datetime.datetime(year=iyear, month=imonth, day=iday)
                ih = int(s_line[13:15])
                imin = int(s_line[16:18])
                isec = int(s_line[19:21])
                sat_sum = int(s_line[33:35])
                iepoc = isec//30+2*imin+2*60*ih

                for irec in range(sat_sum):
                    s_line = f.readline()
                    sat_code = s_line[0:3]
                    if not sat_code in sat_dict:
                        sat_dict[sat_code] = len(sat_dict)
                        sat_list.append(sat_code)
                    idx = sat_dict[sat_code]
                    # Import data
                    if sat_code[0:1] == "G":
                        str_l1 = s_line[3+16*gL_l1_num:17+16*gL_l1_num].strip()
                        str_l2 = s_line[3+16*gL_l2_num:17+16*gL_l2_num].strip()
                        str_p1 = s_line[3+16*gP_l1_num:17+16*gP_l1_num].strip()
                        str_p2 = s_line[3+16*gP_l2_num:17+16*gP_l2_num].strip()
                        l1_bool = False
                        l2_bool = False
                        p1_bool = False
                        p2_bool = False

                        if not str_l1 == "":
                            wave_l1 = float(str_l1)
                            l1_bool = True
                        if not str_l2 == "":
                            wave_l2 = float(str_l2)
                            l2_bool = True
                        if not str_p1 == "":
                            wave_p1 = float(str_p1)
                            p1_bool = True
                        if not str_p2 == "":
                            wave_p2 = float(str_p2)
                            p2_bool = True
                        if l1_bool and l2_bool:
                            # L_stec[idx,iepoc]=6.05/(-0.65)*(vel_light*wave_l2/f_l2_GPS-vel_light*wave_l1/f_l1_GPS)
                            L_stec[idx, iepoc] = (2*math.pow(f_l1_GPS*f_l2_GPS, 2.0))*(vel_light*wave_l1/f_l1_GPS -
                                                                                       vel_light*wave_l2/f_l2_GPS)/(1.0e+16*K*(math.pow(f_l1_GPS, 2.0)-math.pow(f_l2_GPS, 2.0)))
                            # print(sat_code,wave_l1,wave_l2,"->",stec[idx,iepoc])
                            L_stec_bool[idx, iepoc] = True
                        if p1_bool and p2_bool:
                            P_stec[idx, iepoc] = (2*math.pow(f_l1_GPS*f_l2_GPS, 2.0))*(wave_p2-wave_p1)/(
                                1.0e+16*K*(math.pow(f_l1_GPS, 2.0)-math.pow(f_l2_GPS, 2.0)))
                            P_stec_bool[idx, iepoc] = True
                    if sat_code[0:1] == "E":
                        str_l1 = s_line[3+16*eL_l1_num:17+16*eL_l1_num].strip()
                        str_l5 = s_line[3+16*eL_l5_num:17+16*eL_l5_num].strip()
                        str_p1 = s_line[3+16*eP_l1_num:17+16*eP_l1_num].strip()
                        str_p5 = s_line[3+16*eP_l5_num:17+16*eP_l5_num].strip()
                        # print(sat_code,str_l1,str_l2)
                        l1_bool = False
                        l5_bool = False
                        p1_bool = False
                        p5_bool = False

                        if not str_l1 == "":
                            wave_l1 = float(str_l1)
                            l1_bool = True
                        if not str_l5 == "":
                            wave_l5 = float(str_l5)
                            l5_bool = True
                        if not str_p1 == "":
                            wave_p1 = float(str_p1)
                            p1_bool = True
                        if not str_p5 == "":
                            wave_p5 = float(str_p5)
                            p5_bool = True
                        if l1_bool and l5_bool:
                            # L_stec[idx,iepoc]=6.05/(-0.65)*(vel_light*wave_l2/f_l2_GPS-vel_light*wave_l1/f_l1_GPS)
                            L_stec[idx, iepoc] = (2*math.pow(f_l1_Galileo*f_l5_Galileo, 2.0))*(vel_light*wave_l1/f_l1_Galileo -
                                                                                               vel_light*wave_l5/f_l5_Galileo)/(1.0e+16*K*(math.pow(f_l1_Galileo, 2.0)-math.pow(f_l5_Galileo, 2.0)))
                            # print(sat_code,wave_l1,wave_l2,"->",stec[idx,iepoc])
                            L_stec_bool[idx, iepoc] = True
                        if p1_bool and p5_bool:
                            P_stec[idx, iepoc] = (2*math.pow(f_l1_Galileo*f_l5_Galileo, 2.0))*(wave_p5-wave_p1)/(
                                1.0e+16*K*(math.pow(f_l1_Galileo, 2.0)-math.pow(f_l5_Galileo, 2.0)))
                            P_stec_bool[idx, iepoc] = True
                    if sat_code[0:1] == "J":
                        str_l1 = s_line[3+16*jL_l1_num:17+16*jL_l1_num].strip()
                        str_l2 = s_line[3+16*jL_l2_num:17+16*jL_l2_num].strip()
                        str_p1 = s_line[3+16*jP_l1_num:17+16*jP_l1_num].strip()
                        str_p2 = s_line[3+16*jP_l2_num:17+16*jP_l2_num].strip()
                        # print(sat_code,str_l1,str_l2)
                        l1_bool = False
                        l2_bool = False
                        p1_bool = False
                        p2_bool = False

                        if not str_l1 == "":
                            wave_l1 = float(str_l1)
                            l1_bool = True
                        if not str_l2 == "":
                            wave_l2 = float(str_l2)
                            l2_bool = True
                        if not str_p1 == "":
                            wave_p1 = float(str_p1)
                            p1_bool = True
                        if not str_p2 == "":
                            wave_p2 = float(str_p2)
                            p2_bool = True
                        if l1_bool and l2_bool:
                            # L_stec[idx,iepoc]=6.05/(-0.65)*(vel_light*wave_l2/f_l2_GPS-vel_light*wave_l1/f_l1_GPS)
                            L_stec[idx, iepoc] = (2*math.pow(f_l1_QZSS*f_l5_QZSS, 2.0))*(vel_light*wave_l1/f_l1_QZSS -
                                                                                         vel_light*wave_l2/f_l5_QZSS)/(1.0e+16*K*(math.pow(f_l1_QZSS, 2.0)-math.pow(f_l5_QZSS, 2.0)))
                            # print(sat_code,wave_l1,wave_l2,"->",stec[idx,iepoc])
                            L_stec_bool[idx, iepoc] = True
                        if p1_bool and p2_bool:
                            P_stec[idx, iepoc] = (2*math.pow(f_l1_QZSS*f_l5_QZSS, 2.0))*(wave_p2-wave_p1)/(
                                1.0e+16*K*(math.pow(f_l1_QZSS, 2.0)-math.pow(f_l5_QZSS, 2.0)))
                            P_stec_bool[idx, iepoc] = True

    return sat_dict, sat_list, L_stec, L_stec_bool, P_stec, P_stec_bool, p_o, obs_day

# for rinux ver 2.10


def load_observ_file_v2_10(file):
    p_o = np.zeros(3)
    with open(file, "r") as f:
        gL_l1_num = -1
        gL_l2_num = -1
        gP_l1_num = -1
        gP_l2_num = -1

        while True:
            s_line = f.readline()
            if "APPROX POSITION XYZ" in s_line:
                p_o[0] = float(s_line[1:14])
                p_o[1] = float(s_line[15:28])
                p_o[2] = float(s_line[29:42])
            if "TYPES OF OBSERV" in s_line:
                if s_line[4:6].strip() == "":
                    continue
                wave_sum = int(s_line[4:6])

                for i in range(wave_sum):
                    j = i % 9
                    wave = s_line[10+6*j:12+6*j]
                    if wave == "C1":
                        gP_l1_num = i
                    elif wave == "P2":
                        gP_l2_num = i
                    elif wave == "L1":
                        gL_l1_num = i
                    elif wave == "L2":
                        gL_l2_num = i
                    if j == 8:
                        s_line = f.readline()
            if "END OF HEADER" in s_line:
                break
            if not s_line:
                break

        P_stec = np.full((Gmaxsat, 2880), -np.inf, dtype=float)
        P_stec_bool = np.full((Gmaxsat, 2880), False, dtype=bool)
        L_stec = np.full((Gmaxsat, 2880), -np.inf, dtype=float)
        L_stec_bool = np.full((Gmaxsat, 2880), False, dtype=bool)
        sat_dict = {}
        sat_list = []

        for isat in range(Gmaxsat):
            sat_dict["G{i:02d}".format(i=isat)] = isat-1
            sat_list.append("G{i:02d}".format(i=isat))

        while True:
            s_line = f.readline()
            # print(s_line)
            if not s_line:
                break
            if "COMMENT" in s_line:
                continue
            iy_str = s_line[1:3]
            if iy_str != str(year2):
                s_line = f.readline()
            else:
                iyear = int(s_line[1:3])+2000
                imonth = int(s_line[4:6])
                iday = int(s_line[7:9])
                obs_day = datetime.datetime(year=iyear, month=imonth, day=iday)
                ih = int(s_line[10:12])
                imin = int(s_line[13:15])
                isec = int(s_line[16:18])
                sat_sum = int(s_line[30:32])
                # print(imonth, sat_sum)
                iepoc = isec//30+2*imin+2*60*ih
                sat_code_list = []
                for i in range(sat_sum):
                    j = i % 12
                    sat_num = int(s_line[33+3*j:35+3*j])
                    sat_code_list.append("G"+str(sat_num).zfill(2))
                    # print(sat_code_list)
                    if j == 11 and sat_sum > i+1:
                        s_line = f.readline()
                for irec in range(sat_sum):
                    l1_bool = False
                    l2_bool = False
                    p1_bool = False
                    p2_bool = False
                    str_l1 = ""
                    str_l2 = ""
                    str_p1 = ""
                    str_p2 = ""
                    for j in range(-(-wave_sum//5)):
                        s_line = f.readline()
                        # print(s_line.strip())
                        if 5*j <= gL_l1_num and gL_l1_num < 5*(j+1):
                            k = gL_l1_num % 5
                            str_l1 = s_line[1+16*k:14+16*k].strip()
                            if not str_l1 == "":
                                wave_l1 = float(str_l1)
                                l1_bool = True
                        if 5*j <= gL_l2_num and gL_l2_num < 5*(j+1):
                            k = gL_l2_num % 5
                            str_l2 = s_line[1+16*k:14+16*k].strip()
                            if not str_l2 == "":
                                wave_l2 = float(str_l2)
                                l2_bool = True
                        if 5*j <= gP_l1_num and gP_l1_num < 5*(j+1):
                            k = gP_l1_num % 5
                            str_p1 = s_line[1+16*k:14+16*k].strip()
                            if not str_p1 == "":
                                wave_p1 = float(str_p1)
                                p1_bool = True
                        if 5*j <= gP_l2_num and gP_l2_num < 5*(j+1):
                            k = gP_l2_num % 5
                            str_p2 = s_line[1+16*k:14+16*k].strip()
                            if not str_p2 == "":
                                wave_p2 = float(str_p2)
                                p2_bool = True
                    sat_code = sat_code_list[irec]
                    idx = sat_dict[sat_code]
                    if l1_bool and l2_bool:
                        # L_stec[idx,iepoc]=6.05/(-0.65)*(vel_light*wave_l2/f_l2_GPS-vel_light*wave_l1/f_l1_GPS)
                        L_stec[idx, iepoc] = (2*math.pow(f_l1_GPS*f_l2_GPS, 2.0))*(vel_light*wave_l1/f_l1_GPS -
                                                                                   vel_light*wave_l2/f_l2_GPS)/(1.0e+16*K*(math.pow(f_l1_GPS, 2.0)-math.pow(f_l2_GPS, 2.0)))
                        # print(sat_code,wave_l1,wave_l2,"->",stec[idx,iepoc])
                        L_stec_bool[idx, iepoc] = True
                    if p1_bool and p2_bool:
                        P_stec[idx, iepoc] = (2*math.pow(f_l1_GPS*f_l2_GPS, 2.0))*(wave_p2-wave_p1)/(
                            1.0e+16*K*(math.pow(f_l1_GPS, 2.0)-math.pow(f_l2_GPS, 2.0)))
                        P_stec_bool[idx, iepoc] = True

    return sat_dict, sat_list, L_stec, L_stec_bool, P_stec, P_stec_bool, p_o, obs_day


def load_observ_file_v2_12(file):
    p_o = np.zeros(3)
    with open(file, "r") as f:
        # GPS ... A,2
        # Galileo ... 1,5
        # QZSS ... C,5
        L_lA_num = -1
        L_l2_num = -1
        L_l5_num = -1
        L_l1_num = -1
        L_lC_num = -1
        P_lA_num = -1
        P_l2_num = -1
        P_l5_num = -1
        P_l1_num = -1
        P_lC_num = -1

        while True:
            s_line = f.readline()
            if "APPROX POSITION XYZ" in s_line:
                p_o[0] = float(s_line[1:14])
                p_o[1] = float(s_line[15:28])
                p_o[2] = float(s_line[29:42])
            if "TYPES OF OBSERV" in s_line:
                if s_line[4:6].strip() == "":
                    continue
                wave_sum = int(s_line[4:6])

                for i in range(wave_sum):
                    j = i % 9
                    wave = s_line[10+6*j:12+6*j]
                    if wave == "CA":
                        P_lA_num = i
                    elif wave == "P2":
                        P_l2_num = i
                    elif wave == "C5":
                        P_l5_num = i
                    elif wave == "C1":
                        P_l1_num = i
                    elif wave == "CC":
                        P_lC_num = i
                    elif wave == "LA":
                        L_lA_num = i
                    elif wave == "L2":
                        L_l2_num = i
                    elif wave == "L5":
                        L_l5_num = i
                    elif wave == "L1":
                        L_l1_num = i
                    elif wave == "LC":
                        L_lC_num = i
                    if j == 8:
                        s_line = f.readline()
            if "END OF HEADER" in s_line:
                break
            if not s_line:
                break

        P_stec = np.full((maxsat, 2880), -np.inf, dtype=float)
        P_stec_bool = np.full((maxsat, 2880), False, dtype=bool)
        L_stec = np.full((maxsat, 2880), -np.inf, dtype=float)
        L_stec_bool = np.full((maxsat, 2880), False, dtype=bool)
        sat_dict = {}
        sat_list = []

        while True:
            s_line = f.readline()
            if not s_line:
                break
            if "COMMENT" in s_line:
                continue
            iy_str = s_line[1:3]
            if iy_str != str(year2):
                s_line = f.readline()
            else:
                iyear = int(s_line[1:3])+2000
                imonth = int(s_line[4:6])
                iday = int(s_line[7:9])
                obs_day = datetime.datetime(year=iyear, month=imonth, day=iday)
                ih = int(s_line[10:12])
                imin = int(s_line[13:15])
                isec = int(s_line[16:18])
                sat_sum = int(s_line[30:32])
                iepoc = isec//30+2*imin+2*60*ih
                sat_code_list = []
                for i in range(sat_sum):
                    j = i % 12
                    sat_code = s_line[32+3*j:35+3*j]
                    sat_code_list.append(sat_code)
                    if not sat_code in sat_dict:
                        sat_dict[sat_code] = len(sat_dict)
                        sat_list.append(sat_code)
                    # print(sat_code_list)
                    if j == 11 and sat_sum > i+1:
                        s_line = f.readline()
                # print(sat_sum,sat_code_list)
                for irec in range(sat_sum):
                    lA_bool = False
                    l2_bool = False
                    l5_bool = False
                    l1_bool = False
                    lC_bool = False
                    pA_bool = False
                    p2_bool = False
                    p5_bool = False
                    p1_bool = False
                    pC_bool = False
                    str_lA = ""
                    str_l2 = ""
                    str_l5 = ""
                    str_l1 = ""
                    str_lC = ""
                    str_pA = ""
                    str_p2 = ""
                    str_p5 = ""
                    str_p1 = ""
                    str_pC = ""
                    for j in range(-(-wave_sum//5)):
                        s_line = f.readline()
                        # print(s_line.strip())
                        if 5*j <= L_lA_num and L_lA_num < 5*(j+1):
                            k = L_lA_num % 5
                            str_lA = s_line[1+16*k:14+16*k].strip()
                            if not str_lA == "":
                                wave_lA = float(str_lA)
                                lA_bool = True
                        if 5*j <= L_l2_num and L_l2_num < 5*(j+1):
                            k = L_l2_num % 5
                            str_l2 = s_line[1+16*k:14+16*k].strip()
                            if not str_l2 == "":
                                wave_l2 = float(str_l2)
                                l2_bool = True
                        if 5*j <= L_l5_num and L_l5_num < 5*(j+1):
                            k = L_l5_num % 5
                            str_l5 = s_line[1+16*k:14+16*k].strip()
                            if not str_l5 == "":
                                wave_l5 = float(str_l5)
                                l5_bool = True
                        if 5*j <= L_l1_num and L_l1_num < 5*(j+1):
                            k = L_l1_num % 5
                            str_l1 = s_line[1+16*k:14+16*k].strip()
                            if not str_l1 == "":
                                wave_l1 = float(str_l1)
                                l1_bool = True
                        if 5*j <= L_lC_num and L_lC_num < 5*(j+1):
                            k = L_lC_num % 5
                            str_lC = s_line[1+16*k:14+16*k].strip()
                            if not str_lC == "":
                                wave_lC = float(str_lC)
                                lC_bool = True
                        if 5*j <= P_lA_num and P_lA_num < 5*(j+1):
                            k = P_lA_num % 5
                            str_pA = s_line[1+16*k:14+16*k].strip()
                            if not str_pA == "":
                                wave_pA = float(str_pA)
                                pA_bool = True
                        if 5*j <= P_l2_num and P_l2_num < 5*(j+1):
                            k = P_l2_num % 5
                            str_p2 = s_line[1+16*k:14+16*k].strip()
                            if not str_p2 == "":
                                wave_p2 = float(str_p2)
                                p2_bool = True
                        if 5*j <= P_l5_num and P_l5_num < 5*(j+1):
                            k = P_l5_num % 5
                            str_p5 = s_line[1+16*k:14+16*k].strip()
                            if not str_p5 == "":
                                wave_p5 = float(str_p5)
                                p5_bool = True
                        if 5*j <= P_l1_num and P_l1_num < 5*(j+1):
                            k = P_l1_num % 5
                            str_p1 = s_line[1+16*k:14+16*k].strip()
                            if not str_p1 == "":
                                wave_p1 = float(str_p1)
                                p1_bool = True
                        if 5*j <= P_lC_num and P_lC_num < 5*(j+1):
                            k = P_lC_num % 5
                            str_pC = s_line[1+16*k:14+16*k].strip()
                            if not str_pC == "":
                                wave_pC = float(str_pC)
                                pC_bool = True
                    sat_code = sat_code_list[irec]
                    idx = sat_dict[sat_code]
                    if "G" in sat_code:
                        if lA_bool and l2_bool:
                            L_stec[idx, iepoc] = (2*math.pow(f_lA_GPS*f_l2_GPS, 2.0))*(vel_light*wave_lA/f_lA_GPS -
                                                                                       vel_light*wave_l2/f_l2_GPS)/(1.0e+16*K*(math.pow(f_lA_GPS, 2.0)-math.pow(f_l2_GPS, 2.0)))
                            L_stec_bool[idx, iepoc] = True
                        if pA_bool and p2_bool:
                            P_stec[idx, iepoc] = (2*math.pow(f_lA_GPS*f_l2_GPS, 2.0))*(wave_p2-wave_pA)/(
                                1.0e+16*K*(math.pow(f_lA_GPS, 2.0)-math.pow(f_l2_GPS, 2.0)))
                            P_stec_bool[idx, iepoc] = True
                    if "E" in sat_code:
                        if l5_bool and l1_bool:
                            L_stec[idx, iepoc] = (2*math.pow(f_l1_Galileo*f_l5_Galileo, 2.0))*(vel_light*wave_l1/f_l1_Galileo -
                                                                                               vel_light*wave_l5/f_l5_Galileo)/(1.0e+16*K*(math.pow(f_l1_Galileo, 2.0)-math.pow(f_l5_Galileo, 2.0)))
                            L_stec_bool[idx, iepoc] = True
                        if p5_bool and p1_bool:
                            P_stec[idx, iepoc] = (2*math.pow(f_l1_Galileo*f_l5_Galileo, 2.0))*(wave_p5-wave_p1)/(
                                1.0e+16*K*(math.pow(f_l1_Galileo, 2.0)-math.pow(f_l5_Galileo, 2.0)))
                            P_stec_bool[idx, iepoc] = True
                    if "J" in sat_code:
                        if l5_bool and lC_bool:
                            L_stec[idx, iepoc] = (2*math.pow(f_lC_QZSS*f_l5_QZSS, 2.0))*(vel_light*wave_lC/f_lC_QZSS -
                                                                                         vel_light*wave_l5/f_l5_QZSS)/(1.0e+16*K*(math.pow(f_lC_QZSS, 2.0)-math.pow(f_l5_QZSS, 2.0)))
                            L_stec_bool[idx, iepoc] = True
                        if p5_bool and pC_bool:
                            P_stec[idx, iepoc] = (2*math.pow(f_lC_QZSS*f_l5_QZSS, 2.0))*(wave_p5-wave_pC)/(
                                1.0e+16*K*(math.pow(f_lC_QZSS, 2.0)-math.pow(f_l5_QZSS, 2.0)))
                            P_stec_bool[idx, iepoc] = True

    return sat_dict, sat_list, L_stec, L_stec_bool, P_stec, P_stec_bool, p_o, obs_day

# ************************************************************
# [yy]n,[yy]g,[yy]l,[yy]q reader


# for RINEX 2.10 / 2.12 for GPS
# if code like "01" ... code = 0
#         like "G01" ... code = 1
def load_navi_file_v2(file, obs_day, code=0):
    base_dt = obs_day
    oel = np.zeros((maxrec, 28))  # storing navigation data
    idsat = np.zeros(maxrec, dtype=int)
    tiempo = np.zeros(maxrec)
    i1stsat = [[] for i in range(Gmaxsat)]

    with open(file, mode='r', errors='replace') as f_n:
        #
        # reading header of navigation file
        #
        while True:
            s_line = f_n.readline()
            # if 'RINEX VERSION' in s_line:
            #     ver = s_line[5:9]
            if 'END OF HEADER' in s_line:
                break
            if not s_line:
                break
        #
        # reading data of navigation file
        #
        krec = 0
        s_line = f_n.readline()
        while True:
            idsat[krec] = int(s_line[0+code:2+code])
            i1stsat[int(s_line[0+code:2+code])].append(krec)
            iy = int(s_line[3+code:5+code])+2000
            imonth = int(s_line[6+code:8+code])
            iday = int(s_line[9+code:11+code])
            ih = int(s_line[12+code:14+code])
            imin = int(s_line[15+code:17+code])
            sec = float(s_line[18+code:22+code])
            dt2 = datetime.datetime(
                year=iy, month=imonth, day=iday, hour=ih, minute=imin, second=int(sec))
            td = dt2-base_dt
            tiempo[krec] = td.total_seconds()/3600.0
            s_line = f_n.readline()
            for irec in range(7):
                ii = irec*4
                if irec != 6:
                    oel[krec][ii] = float(s_line[3+code:3+15+code]) * \
                        math.pow(10, int(s_line[19+code:19+3+code]))
                    oel[krec][ii+1] = float(s_line[3+19+code:3+19+15+code]) * \
                        math.pow(10, int(s_line[19+19+code:19+19+3+code]))
                    oel[krec][ii+2] = float(s_line[3+19*2+code:3+19*2+15+code]) * \
                        math.pow(10, int(s_line[19+19*2+code:19+19*2+3+code]))
                    oel[krec][ii+3] = float(s_line[3+19*3+code:3+19*3+15+code]) * \
                        math.pow(10, int(s_line[19+19*3+code:19+19*3+3+code]))
                else:
                    oel[krec][ii] = float(s_line[3+code:3+15+code]) * \
                        math.pow(10, int(s_line[19+code:19+3+code]))
                    oel[krec][ii+1] = 0.0
                    oel[krec][ii+2] = 0.0
                    oel[krec][ii+3] = 0.0
                s_line = f_n.readline()
            krec += 1
            if not s_line:
                break

        return idsat, tiempo, oel, i1stsat
# GPS,Galileo,QZSS # with 3-4km at 20000km Alt. ~100m at 500km Alt.


def make_orbit_data(isat: int, tiempo, oel, i1stsat) -> np.ndarray:

    # orbit[i,j,k] = epoc i, sat j, coordinate k position
    orbit = np.full((2880, 3), 0.0, dtype=float)

    if len(i1stsat[isat]) > 0:
        base = 0
        for iepoc in range(2880):
            ttime = iepoc/120.0
            time0 = tiempo[i1stsat[isat][base]]
            try:
                xyz = gtxyz(ttime, time0, oel[i1stsat[isat][base]])
                for k in range(3):
                    orbit[iepoc, k] = xyz[k]
                base = 0
            except:
                if base+1 < len(i1stsat[isat]):
                    base += 1
                    iepoc -= 1
                    continue
                else:
                    pass

        return orbit
    else:
        return orbit

# isat is the featured satelite id


def make_tomo_file(path_op, L_valid, mod_L_stec, P_stec, L_stec, L_stec_bool, P_stec_bool, orbit_data, orbit_bool, p_o, isat, sat_num, f_l1, f_l2, sat_id):
    with open(path_op, "w") as f_nd:
        flag = True
        for iepoc in range(2880):
            if (L_valid[isat, iepoc] == 1 or L_valid[isat, iepoc] == 2) and (L_stec_bool[isat, iepoc] and P_stec_bool[isat, iepoc]):
                ttime = iepoc/120.0
                if flag:
                    f_nd.write("# PRN {sat}\n".format(sat=sat_id))
                    f_nd.write("# \n")
                    f_nd.write("# RINEX VER G_3.02\n")
                    f_nd.write("# FILE {s:>4}{d:03d}0.{y2:02d}o {s:>4}{d:03d}0.{y2:02d}n\n".format(
                        s=sta, d=day, y2=year2))
                    f_nd.write("# \n")
                    f_nd.write("# RUN BY\n")
                    f_nd.write("# PROGRAM {p}\n".format(p=version))
                    f_nd.write("# UTCTIME {t}\n".format(
                        t=datetime.datetime.now()))
                    f_nd.write("# \n")
                    f_nd.write("# WAVE FREQUENCY\n")
                    f_nd.write("# L1 {l1}\n".format(l1=f_l1))
                    f_nd.write("# L2 {l2}\n".format(l2=f_l2))
                    f_nd.write("# \n")
                    f_nd.write(
                        "# CYCLE SLIP THRESHOLD {th:8.6f}\n".format(th=threshold))
                    f_nd.write("# \n")
                    f_nd.write("# 1. TIME [UTC]\n")
                    f_nd.write("# 2. MODIFIED L CODE STEC [TECU]\n")
                    f_nd.write("# 3. P CODE STEC [TECU]\n")
                    f_nd.write("# 4. L CODE STEC [TECU]\n")
                    f_nd.write("# 5. SATELITE POSITION [m]\n")
                    f_nd.write("# 6. RECEIVER POSITION [m]\n")
                    f_nd.write("# \n")
                    f_nd.write("# END OF HEADER\n")
                    flag = False
                if orbit_bool[iepoc, sat_num]:
                    mL_stec = mod_L_stec[isat, iepoc]
                    rL_stec = L_stec[isat, iepoc]
                    rP_stec = P_stec[isat, iepoc]
                    rec = XYZ()
                    rec.set(p_o[0], p_o[1], p_o[2])
                    sat = XYZ()
                    sat.set(orbit_data[iepoc, sat_num, 0], orbit_data[iepoc,
                            sat_num, 1], orbit_data[iepoc, sat_num, 2])
                    Z = r_zenith(rec, sat)
                    f_nd.write('{t:07.4f} {mls:09.4f} {ps:09.4f} {ls:09.4f} {x:014.4f} {y:014.4f} {z:014.4f} {p_x:014.4f} {p_y:014.4f} {p_z:014.4f} {zenith:010.7f}\n'.format(
                        t=ttime, mls=mL_stec, ps=rP_stec, ls=rL_stec, x=orbit_data[iepoc, sat_num, 0],
                        y=orbit_data[iepoc, sat_num, 1], z=orbit_data[iepoc, sat_num, 2], p_x=p_o[0], p_y=p_o[1], p_z=p_o[2], zenith=90.0-Z*180./math.pi))


# rnx to abs stec
now_station = ""


def rnx2mdf_v3_02(year4: int, day: int, stations: list):
    year2 = year4 % 100
    md = datetime.date(year4, 1, 1)+datetime.timedelta(day-1)

    obs_day = datetime.datetime(year4, md.month, md.day)

    G_orbit_data = np.full((2880, Gmaxsat, 3), 0.0, dtype=float)
    G_orbit_bool = np.full((2880, Gmaxsat), False, dtype=bool)
    G_finish_check = np.full((Gmaxsat), False, dtype=bool)
    E_orbit_data = np.full((2880, Emaxsat, 3), 0.0, dtype=float)
    E_orbit_bool = np.full((2880, Emaxsat), False, dtype=bool)
    E_finish_check = np.full((Emaxsat), False, dtype=bool)
    J_orbit_data = np.full((2880, Jmaxsat, 3), 0.0, dtype=float)
    J_orbit_bool = np.full((2880, Jmaxsat), False, dtype=bool)
    J_finish_check = np.full((Jmaxsat), False, dtype=bool)

    for sta in stations:
        # print(E_orbit_data[307,9])
        # About 15 sec for each station (ver 1.0)
        # About 7 sec for each station (ver 1.2)
        # About 15 sec for each station (ver 2.0)
        # Read observation data

        path_o = path_tec + \
            "{s}{d:03d}0.{y2:02}o".format(y=year4, y2=year2, d=day, s=sta)
        # L_stec,L_stec_bool,P_stec,P_stec_bool
        # [i][j] ... index i(sat_dict[i]) satelite, epoch j data
        # sat_bool[i] ... id i satelite
        # sat_dict[id] ... L_stec index of sat id
        # sat_list[i] ... index i satelite
        sat_dict, sat_list, L_stec, L_stec_bool, P_stec, P_stec_bool, p_o, obs_day = load_observ_file_v3_02(
            path_o)

        dt1 = time.time()-dt_now
        # print(dt1)
        # Read navigation data
        # if has any False
        if not np.all(G_finish_check):
            GPS_file = path_tec + \
                "{s}{d:03}0.{y2:02}n".format(s=sta, d=day, y2=year2)

            if os.path.exists(GPS_file):
                # idsat[k] ... k th record satelite id
                # oel[k][0-27] ... k th record
                # tiempo[k] ... k th record time from obs_day 00:00:00UTC
                # i1stsat[k] ... list of k th satelite record number
                G_idsat, G_tiempo, G_oel, G_i1stsat = load_navi_file(
                    GPS_file, obs_day)

            for isat in range(1, Gmaxsat):
                # if dont finish
                if not G_finish_check[isat]:
                    Gor_data = make_orbit_data(
                        isat, G_tiempo, G_oel, G_i1stsat)
                    for iepoc in range(2880):
                        if sum(abs(Gor_data[iepoc, :])) > 200.0:
                            G_orbit_data[iepoc, isat,
                                         :] = Gor_data[iepoc, :]
                            G_orbit_bool[iepoc, isat] = True

                if np.all(G_orbit_bool[:, isat]):
                    G_finish_check[isat] = True

        if not np.all(E_finish_check):
            Galileo_file = path_tec + \
                "{s}{d:03}0.{y2:02}l".format(s=sta, d=day, y2=year2)

            if os.path.exists(Galileo_file):
                # idsat[k] ... k th record satelite id
                # oel[k][0-27] ... k th record
                # tiempo[k] ... k th record time from obs_day 00:00:00UTC
                # i1stsat[k] ... list of k th satelite record number
                E_idsat, E_tiempo, E_oel, E_i1stsat = load_navi_file(
                    Galileo_file, obs_day)

            for isat in range(1, Emaxsat):
                if not E_finish_check[isat]:
                    Eor_data = make_orbit_data(
                        isat, E_tiempo, E_oel, E_i1stsat)
                    for iepoc in range(2880):
                        if sum(abs(Eor_data[iepoc, :])) > 200.0:
                            E_orbit_data[iepoc, isat,
                                         :] = Eor_data[iepoc, :]
                            E_orbit_bool[iepoc, isat] = True

                if np.all(E_orbit_bool[:, isat]):
                    E_finish_check[isat] = True

        if not np.all(J_finish_check):
            QZSS_file = path_tec + \
                "{s}{d:03}0.{y2:02}q".format(s=sta, d=day, y2=year2)

            if os.path.exists(QZSS_file):
                # idsat[k] ... k th record satelite id
                # oel[k][0-27] ... k th record
                # tiempo[k] ... k th record time from obs_day 00:00:00UTC
                # i1stsat[k] ... list of k th satelite record number
                J_idsat, J_tiempo, J_oel, J_i1stsat = load_navi_file(
                    QZSS_file, obs_day)

            for isat in range(1, Jmaxsat):
                if not J_finish_check[isat]:
                    Jor_data = make_orbit_data(
                        isat, J_tiempo, J_oel, J_i1stsat)
                    for iepoc in range(2880):
                        if sum(abs(Jor_data[iepoc, :])) > 200.0:
                            J_orbit_data[iepoc, isat,
                                         :] = Jor_data[iepoc, :]
                            J_orbit_bool[iepoc, isat] = True

                if np.all(J_orbit_bool[:, isat]):
                    J_finish_check[isat] = True

        # orbit[i,j,k] = epoc i, sat j, coordinate k position
        dt2 = time.time()-dt_now
        # print(dt2)
        # print(L_stec)
        # a=input()
        # L_data_set[i,j]...number of set, sat i, epoch j
        # L_data_len[i] ... length of set i
        # L_data_sat[i] ... satelite of set i
        # L_data_begin[i] ... first epoch of set i
        L_data_set = np.full((maxsat, 2880), -1, dtype=int)
        L_data_len = np.full((2000), 0, dtype=int)
        L_data_sat_idx = np.full((2000), 0, dtype=int)
        L_data_begin = np.full((2000), 0, dtype=int)
        # 0 ... �?ータな�?
        # 1 ... L,P, not start
        # 2 ... L,P, start
        # 3 ... L, not start
        # 4 ... L, start
        L_info = np.full((maxsat, 2880), 0, dtype=int)
        # 0 ... �?ータ無効
        # 1 ... �?ータ有効 but 計算には�?れな�?
        # 2 ... �?ータ有効 and 計算に�?れる
        L_valid = np.full((maxsat, 2880), 0, dtype=int)
        Zs = np.full((maxsat, 2880), 0.0, dtype=float)
        n_data = -1

        for isat in range(len(sat_list)):
            sat_id = sat_list[isat]
            # print(sat_id)
            rec = XYZ()
            rec.set(p_o[0], p_o[1], p_o[2])
            for iepoc in range(2880):
                if "G" in sat_id:
                    if not G_orbit_bool[iepoc, int(sat_id[1:3])]:
                        L_info[isat, iepoc] = 0
                        continue
                    sat_x = G_orbit_data[iepoc, int(sat_id[1:3]), 0]
                    sat_y = G_orbit_data[iepoc, int(sat_id[1:3]), 1]
                    sat_z = G_orbit_data[iepoc, int(sat_id[1:3]), 2]
                    sat = XYZ()
                    sat.set(sat_x, sat_y, sat_z)
                    iZ = r_zenith(rec, sat)
                elif "E" in sat_id:
                    if not E_orbit_bool[iepoc, int(sat_id[1:3])]:
                        L_info[isat, iepoc] = 0
                        continue
                    sat_x = E_orbit_data[iepoc, int(sat_id[1:3]), 0]
                    sat_y = E_orbit_data[iepoc, int(sat_id[1:3]), 1]
                    sat_z = E_orbit_data[iepoc, int(sat_id[1:3]), 2]
                    sat = XYZ()
                    sat.set(sat_x, sat_y, sat_z)
                    iZ = r_zenith(rec, sat)
                elif "J" in sat_id:
                    if not J_orbit_bool[iepoc, int(sat_id[1:3])]:
                        L_info[isat, iepoc] = 0
                        continue
                    sat_x = J_orbit_data[iepoc, int(sat_id[1:3]), 0]
                    sat_y = J_orbit_data[iepoc, int(sat_id[1:3]), 1]
                    sat_z = J_orbit_data[iepoc, int(sat_id[1:3]), 2]
                    sat = XYZ()
                    sat.set(sat_x, sat_y, sat_z)
                    iZ = r_zenith(rec, sat)
                Zs[isat][iepoc] = iZ*180./math.pi
                if iepoc == 0:
                    # L,P両方�?ータがあ�? -> 新しいブロ�?クの始ま�?
                    if L_stec_bool[isat][iepoc] and P_stec_bool[isat][iepoc]:
                        if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                            L_info[isat][iepoc] = 2
                    # Lのみある -> 4
                    elif L_stec_bool[isat][iepoc]:
                        if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                            L_info[isat][iepoc] = 4
                    # どちらも初めの�?ータがな�? -> �?ータな�?
                    else:
                        L_info[isat][iepoc] = 0
                else:
                    now_L_stec = L_stec[isat, iepoc]
                    # ep=iepocのP,L�?ータあり
                    if L_stec_bool[isat][iepoc] and P_stec_bool[isat][iepoc]:
                        ibefore = -1
                        for jbefore in range(1, min(null_threshold+2, iepoc+1)):
                            if L_info[isat][iepoc-jbefore] > 0:
                                ibefore = jbefore
                                break
                        # 前に有効な�?ータな�? -> start
                        if ibefore == -1:
                            if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                L_info[isat][iepoc] = 2
                        # 有効な�?ータあり
                        else:
                            before_L_stec = L_stec[isat, iepoc-ibefore]
                            # サイクルスリ�?�? -> start
                            if abs(now_L_stec-before_L_stec) > threshold:
                                if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                    L_info[isat][iepoc] = 2
                            # 非サイクルスリ�?�? -> continue
                            else:
                                if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                    L_info[isat][iepoc] = 1
                    # ep=iepocのL�?ータのみ
                    elif L_stec_bool[isat][iepoc]:
                        ibefore = -1
                        for jbefore in range(1, min(null_threshold+2, iepoc)):
                            if L_info[isat][iepoc-jbefore] > 0:
                                ibefore = jbefore
                        # 前に有効な�?ータな�? -> start
                        if ibefore == -1:
                            if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                L_info[isat][iepoc] = 4
                        # 有効な�?ータあり
                        else:
                            before_L_stec = L_stec[isat, iepoc-ibefore]
                            # サイクルスリ�?�? -> start
                            if abs(now_L_stec-before_L_stec) > threshold:
                                if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                    L_info[isat][iepoc] = 4
                            # 非サイクルスリ�?�? -> continue
                            else:
                                if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                    L_info[isat][iepoc] = 3
                    # ep=1のL�?ータな�?
                    else:
                        L_info[isat][iepoc] = 0

        # for isat in range(0, 10):
        #     for iepoc in range(2880):
        #         if L_info[isat, iepoc] == 0:
        #             print(sat_list[isat], iepoc, ":", L_stec_bool[isat, iepoc], P_stec_bool[isat, iepoc], L_stec[isat,
        #                                                                                                          iepoc], P_stec[isat, iepoc], L_info[isat, iepoc], L_valid[isat, iepoc], Zs[isat, iepoc])
        #         if L_info[isat, iepoc] == 1 or L_info[isat, iepoc] == 3:
        #             print("*", sat_list[isat], iepoc, ":", L_stec_bool[isat, iepoc], P_stec_bool[isat, iepoc], L_stec[isat,
        #                                                                                                               iepoc], P_stec[isat, iepoc], L_info[isat, iepoc], L_valid[isat, iepoc], Zs[isat, iepoc])
        #         if L_info[isat, iepoc] == 2 or L_info[isat, iepoc] == 4:
        #             print("***", sat_list[isat], iepoc, ":", L_stec_bool[isat, iepoc], P_stec_bool[isat, iepoc], L_stec[isat,
        #                                                                                                                 iepoc], P_stec[isat, iepoc], L_info[isat, iepoc], L_valid[isat, iepoc], Zs[isat, iepoc], "***")
        #     input()

        # Valid_Data_Length = 60

        for isat in range(maxsat):
            # 直前�?�L_info=2
            before_slip = 0
            count = 0
            for iepoc in range(2880):
                # �?ータな�?
                # �?ータ数は増やさな�?
                if L_info[isat][iepoc] == 0:
                    pass
                # �?ータあり、スタートでな�?
                # �?ータ数�?け増や�?
                elif L_info[isat][iepoc] == 1:
                    count += 1
                # �?ータあり、スター�?
                elif L_info[isat][iepoc] == 2 or L_info[isat][iepoc] == 4:
                    # 直前�?�L_info=2からの�?ータ数が規定値以�? -> 有効
                    if count >= Valid_Data_Length:
                        for jepoc in range(before_slip, iepoc):
                            if L_info[isat][jepoc] == 1 or L_info[isat][jepoc] == 2:
                                L_valid[isat][jepoc] = 2
                            elif L_info[isat][jepoc] == 3 or L_info[isat][jepoc] == 4:
                                L_valid[isat][jepoc] = 1
                    before_slip = iepoc
                    count = 0
                if iepoc == 2879:
                    if count >= Valid_Data_Length:
                        for jepoc in range(before_slip, 2880):
                            if L_info[isat][jepoc] == 1 or L_info[isat][jepoc] == 2:
                                L_valid[isat][jepoc] = 2
                            elif L_info[isat][jepoc] == 3 or L_info[isat][jepoc] == 4:
                                L_valid[isat][jepoc] = 1
        for isat in range(maxsat):
            for iepoc in range(2880):
                # �?ータな�?
                if L_info[isat][iepoc] == 0:
                    pass
                # スタートでな�?
                elif L_info[isat][iepoc] == 1 or L_info[isat][iepoc] == 3:
                    if L_valid[isat][iepoc]:
                        L_data_set[isat][iepoc] = n_data
                        L_data_len[n_data] += 1
                # スター�?
                elif L_info[isat][iepoc] == 2 or L_info[isat][iepoc] == 4:
                    # 有効
                    if L_valid[isat][iepoc]:
                        n_data += 1
                        L_data_set[isat][iepoc] = n_data
                        L_data_sat_idx[n_data] = isat
                        L_data_begin[n_data] = iepoc
                        L_data_len[n_data] += 1

        n_data += 1
        dt3 = time.time()-dt_now
        # print(L_data_sat)
        P_O = XYZ()
        P_O.set(p_o[0], p_o[1], p_o[2])
        B_data = np.full((n_data), 0.0, dtype=float)
        for idata in range(n_data):
            # print(idata, "/", n_data)
            isat_idx = L_data_sat_idx[idata]
            iblock_bgn = L_data_begin[idata]
            # 最後�?�ブロ�?ク
            if idata == n_data-1:
                nextblock_bgn = 2880
            # 同じ衛星の次のブロ�?クのはじまりまで
            elif isat_idx == L_data_sat_idx[idata+1]:
                nextblock_bgn = L_data_begin[idata+1]
            # 別の衛星
            else:
                nextblock_bgn = 2880
            # print(idata, "/", n_data, sat_list[isat_idx])
            lower = 0.0
            upper = 0.0
            if "G" in sat_list[isat_idx]:
                isat = int(sat_list[isat_idx][1:3])
                for iepoc in range(iblock_bgn, nextblock_bgn):
                    # �?ータな�?
                    if L_info[isat_idx][iepoc] == 0:
                        pass
                    elif L_info[isat_idx][iepoc] == 1 or L_info[isat_idx][iepoc] == 2:
                        if L_valid[isat_idx][iepoc]:
                            jORBIT = XYZ()
                            jORBIT.set(G_orbit_data[iepoc, isat, 0],
                                       G_orbit_data[iepoc, isat, 1], G_orbit_data[iepoc, isat, 2])
                            lL = L_stec[isat_idx, iepoc]
                            lP = P_stec[isat_idx, iepoc]
                            j_ipp = specify_H(P_O, jORBIT, H_ipp*1.0e+3)
                            lZ = zenith(P_O, j_ipp)
                            lower += np.power(np.sin(lZ), 2)
                            upper += (lP-lL)*np.power(np.sin(lZ), 2)
                    else:
                        pass
                B = upper/lower
                B_data[idata] = B

            elif "E" in sat_list[isat_idx]:
                isat = int(sat_list[isat_idx][1:3])
                for iepoc in range(iblock_bgn, nextblock_bgn):
                    # �?ータな�?
                    if L_info[isat_idx][iepoc] == 0:
                        pass
                    elif L_info[isat_idx][iepoc] == 1 or L_info[isat_idx][iepoc] == 2:
                        if L_valid[isat_idx][iepoc]:
                            jORBIT = XYZ()
                            jORBIT.set(E_orbit_data[iepoc, isat, 0],
                                       E_orbit_data[iepoc, isat, 1], E_orbit_data[iepoc, isat, 2])
                            lL = L_stec[isat_idx, iepoc]
                            lP = P_stec[isat_idx, iepoc]
                            j_ipp = specify_H(P_O, jORBIT, H_ipp*1.0e+3)
                            lZ = zenith(P_O, j_ipp)
                            lower += np.power(np.sin(lZ), 2)
                            upper += (lP-lL)*np.power(np.sin(lZ), 2)
                    else:
                        pass
                B = upper/lower
                B_data[idata] = B
            elif "J" in sat_list[isat_idx]:
                isat = int(sat_list[isat_idx][1:3])
                for iepoc in range(iblock_bgn, nextblock_bgn):
                    # �?ータな�?
                    if L_info[isat_idx][iepoc] == 0:
                        pass
                    elif L_info[isat_idx][iepoc] == 1 or L_info[isat_idx][iepoc] == 2:
                        if L_valid[isat_idx][iepoc]:
                            jORBIT = XYZ()
                            jORBIT.set(J_orbit_data[iepoc, isat, 0],
                                       J_orbit_data[iepoc, isat, 1], J_orbit_data[iepoc, isat, 2])
                            lL = L_stec[isat_idx, iepoc]
                            lP = P_stec[isat_idx, iepoc]
                            j_ipp = specify_H(P_O, jORBIT, H_ipp*1.0e+3)
                            lZ = zenith(P_O, j_ipp)
                            lower += np.power(np.sin(lZ), 2)
                            upper += (lP-lL)*np.power(np.sin(lZ), 2)
                    else:
                        pass
                B = upper/lower
                B_data[idata] = B

        # for isat in range(0, 10):
        #     for iepoc in range(2880):
        #         if L_info[isat, iepoc] == 0:
        #             print(sat_list[isat], iepoc, ":", L_stec_bool[isat, iepoc], P_stec_bool[isat, iepoc], L_stec[isat,
        #                                                                                                          iepoc], P_stec[isat, iepoc], L_info[isat, iepoc], L_valid[isat, iepoc], Zs[isat, iepoc])
        #         if L_info[isat, iepoc] == 1 or L_info[isat, iepoc] == 3:
        #             print("*", sat_list[isat], iepoc, ":", L_stec_bool[isat, iepoc], P_stec_bool[isat, iepoc], L_stec[isat,
        #                                                                                                               iepoc], P_stec[isat, iepoc], L_info[isat, iepoc], L_valid[isat, iepoc], Zs[isat, iepoc])
        #         if L_info[isat, iepoc] == 2 or L_info[isat, iepoc] == 4:
        #             print("***", sat_list[isat], iepoc, ":", L_stec_bool[isat, iepoc], P_stec_bool[isat, iepoc], L_stec[isat,
        #                                                                                                                 iepoc], P_stec[isat, iepoc], L_info[isat, iepoc], L_valid[isat, iepoc], Zs[isat, iepoc], "***")
        #     input()
        # for idata in range(n_data):
        #     print(idata, sat_list[L_data_sat_idx[idata]], L_data_begin[idata])
        # input()
        dt4 = time.time()-dt_now
        # print(dt4)
        # mod_L_stec[i][j] ... satelite i, epoch j
        mod_L_stec = np.full((maxsat, 2880), 0.0, dtype=float)

        for idata in range(n_data):
            isat = L_data_sat_idx[idata]
            iblock_bgn = L_data_begin[idata]
            # 最後�?�ブロ�?ク
            if idata == n_data-1:
                nextblock_bgn = 2880
            # 同じ衛星の次のブロ�?クのはじまりまで
            elif isat == L_data_sat_idx[idata+1]:
                nextblock_bgn = L_data_begin[idata+1]
            # 別の衛星
            else:
                nextblock_bgn = 2880
            # print(idata, ":", sat_list[L_data_sat_idx[idata]],
            #       L_data_begin[idata], nextblock_bgn, L_data_len[idata], B_data[idata], isat, "->", L_data_sat_idx[idata+1])
            iB = B_data[idata]
            for iepoc in range(iblock_bgn, nextblock_bgn):
                # �?ータな�?
                if L_info[isat][iepoc] == 0:
                    pass
                else:
                    if L_valid[isat][iepoc] > 0:
                        mod_L_stec[isat, iepoc] = L_stec[isat, iepoc]+iB

        for isat in range(len(sat_list)):
            path_nw = "{md}/{c}/{y:04}/{d:03d}/{sat}".format(
                md=mdf_folder, y=year4, d=day, sat=sat_list[isat], c=country)
            path_nd = path_nw + "/{s:4}.mdf".format(s=sta)
            sat_num = int(sat_list[isat][1:3])
            if "G" in sat_list[isat]:
                os.makedirs(path_nw, exist_ok=True)
                f_l1 = f_l1_GPS
                f_l2 = f_l2_GPS
                # print(sat_list[isat],sat_num,G_orbit_bool[:,sat_num])
                # isat ... index of L_stec
                # sat_num ... number id of satelite
                make_tomo_file(path_nd, L_valid, mod_L_stec, P_stec, L_stec, L_stec_bool, P_stec_bool,
                               G_orbit_data, G_orbit_bool, p_o, isat, sat_num, f_l1, f_l2, sat_list[isat])
            elif "E" in sat_list[isat]:
                os.makedirs(path_nw, exist_ok=True)
                f_l1 = f_l1_Galileo
                f_l2 = f_l5_Galileo
                make_tomo_file(path_nd, L_valid, mod_L_stec, P_stec, L_stec, L_stec_bool, P_stec_bool,
                               E_orbit_data, E_orbit_bool, p_o, isat, sat_num, f_l1, f_l2, sat_list[isat])
            elif "J" in sat_list[isat]:
                os.makedirs(path_nw, exist_ok=True)
                f_l1 = f_l1_QZSS
                f_l2 = f_l5_QZSS
                make_tomo_file(path_nd, L_valid, mod_L_stec, P_stec, L_stec, L_stec_bool, P_stec_bool,
                               J_orbit_data, J_orbit_bool, p_o, isat, sat_num, f_l1, f_l2, sat_list[isat])

        print("{y:04} {d:03} {s} end : {t:09.3f} ( {d1:09.3f} {d2:09.3f} {d3:09.3f} {d4:09.3f} )".format(
            y=year4, d=day, s=sta, t=time.time()-dt_now, d1=dt1, d2=dt2, d3=dt3, d4=dt4))


def rnx2mdf_v2_10(year4: int, day: int, stations: list):
    year2 = year4 % 100
    md = datetime.date(year4, 1, 1)+datetime.timedelta(day-1)

    obs_day = datetime.datetime(year4, md.month, md.day)

    G_orbit_data = np.full((2880, Gmaxsat, 3), 0.0, dtype=float)
    G_orbit_bool = np.full((2880, Gmaxsat), False, dtype=bool)
    G_finish_check = np.full((Gmaxsat), False, dtype=bool)

    for sta in stations:
        # print(E_orbit_data[307,9])
        # About 15 sec for each station (ver 1.0)
        # About 7 sec for each station (ver 1.2)
        # About 15 sec for each station (ver 2.0)
        # Read observation data
        path_o = path_tec + \
            "{s}{d:03d}0.{y2:02}o".format(y=year4, y2=year2, d=day, s=sta)
        # L_stec,L_stec_bool,P_stec,P_stec_bool
        # [i][j] ... index i(sat_dict[i]) satelite, epoch j data
        # sat_bool[i] ... id i satelite
        # sat_dict[id] ... L_stec index of sat id
        # sat_list[i] ... index i satelite
        sat_dict, sat_list, L_stec, L_stec_bool, P_stec, P_stec_bool, p_o, obs_day = load_observ_file_v2_10(
            path_o)

        dt1 = time.time()-dt_now
        # Read navigation data
        # if has any False
        if not np.all(G_finish_check):
            GPS_file = path_tec + \
                "{s}{d:03}0.{y2:02}n".format(s=sta, d=day, y2=year2)

            if os.path.exists(GPS_file):
                # idsat[k] ... k th record satelite id
                # oel[k][0-27] ... k th record
                # tiempo[k] ... k th record time from obs_day 00:00:00UTC
                # i1stsat[k] ... list of k th satelite record number
                G_idsat, G_tiempo, G_oel, G_i1stsat = load_navi_file_v2(
                    GPS_file, obs_day)

            for isat in range(1, Gmaxsat):
                # if dont finish
                if not G_finish_check[isat]:
                    Gor_data = make_orbit_data(
                        isat, G_tiempo, G_oel, G_i1stsat)
                    for iepoc in range(2880):
                        if sum(abs(Gor_data[iepoc, :])) > 200.0:
                            G_orbit_data[iepoc, isat,
                                         :] = Gor_data[iepoc, :]
                            G_orbit_bool[iepoc, isat] = True

                if np.all(G_orbit_bool[:, isat]):
                    G_finish_check[isat] = True
        # orbit[i,j,k] = epoc i, sat j, coordinate k position
        dt2 = time.time()-dt_now
        # print(L_stec)
        # a=input()
        # L_data_set[i,j]...number of set, sat i, epoch j
        # L_data_len[i] ... length of set i
        # L_data_sat[i] ... satelite of set i
        # L_data_begin[i] ... first epoch of set i
        L_data_set = np.full((Gmaxsat, 2880), -1, dtype=int)
        L_data_len = np.full((500), 0, dtype=int)
        L_data_sat_idx = np.full((500), 0, dtype=int)
        L_data_begin = np.full((500), 0, dtype=int)
        # 0 ... �?ータな�?
        # 1 ... L,P, not start
        # 2 ... L,P, start
        # 3 ... L, not start
        # 4 ... L, start
        L_info = np.full((Gmaxsat, 2880), 0, dtype=int)
        # 0 ... �?ータ無効
        # 1 ... �?ータ有効 but 計算には�?れな�?
        # 2 ... �?ータ有効 and 計算に�?れる
        L_valid = np.full((Gmaxsat, 2880), 0, dtype=int)
        Zs = np.full((Gmaxsat, 2880), 0.0, dtype=float)
        n_data = -1

        for isat in range(len(sat_list)):
            sat_id = sat_list[isat]
            # print(sat_id)
            rec = XYZ()
            rec.set(p_o[0], p_o[1], p_o[2])
            for iepoc in range(2880):
                if not G_orbit_bool[iepoc, int(sat_id[1:3])]:
                    L_info[isat, iepoc] = 0
                    continue
                sat_x = G_orbit_data[iepoc, int(sat_id[1:3]), 0]
                sat_y = G_orbit_data[iepoc, int(sat_id[1:3]), 1]
                sat_z = G_orbit_data[iepoc, int(sat_id[1:3]), 2]
                sat = XYZ()
                sat.set(sat_x, sat_y, sat_z)
                iZ = r_zenith(rec, sat)
                Zs[isat][iepoc] = iZ*180./math.pi
                if iepoc == 0:
                    # L,P両方�?ータがあ�? -> 新しいブロ�?クの始ま�?
                    if L_stec_bool[isat][iepoc] and P_stec_bool[isat][iepoc]:
                        if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                            L_info[isat][iepoc] = 2
                    # Lのみある -> 4
                    elif L_stec_bool[isat][iepoc]:
                        if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                            L_info[isat][iepoc] = 4
                    # どちらも初めの�?ータがな�? -> �?ータな�?
                    else:
                        L_info[isat][iepoc] = 0
                else:
                    now_L_stec = L_stec[isat, iepoc]
                    # ep=iepocのP,L�?ータあり
                    if L_stec_bool[isat][iepoc] and P_stec_bool[isat][iepoc]:
                        ibefore = -1
                        for jbefore in range(1, min(null_threshold+2, iepoc+1)):
                            if L_info[isat][iepoc-jbefore] > 0:
                                ibefore = jbefore
                                break
                        # 前に有効な�?ータな�? -> start
                        if ibefore == -1:
                            if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                L_info[isat][iepoc] = 2
                        # 有効な�?ータあり
                        else:
                            before_L_stec = L_stec[isat, iepoc-ibefore]
                            # サイクルスリ�?�? -> start
                            if abs(now_L_stec-before_L_stec) > threshold:
                                if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                    L_info[isat][iepoc] = 2
                            # 非サイクルスリ�?�? -> continue
                            else:
                                if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                    L_info[isat][iepoc] = 1
                    # ep=iepocのL�?ータのみ
                    elif L_stec_bool[isat][iepoc]:
                        ibefore = -1
                        for jbefore in range(1, min(null_threshold+2, iepoc)):
                            if L_info[isat][iepoc-jbefore] > 0:
                                ibefore = jbefore
                        # 前に有効な�?ータな�? -> start
                        if ibefore == -1:
                            if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                L_info[isat][iepoc] = 4
                        # 有効な�?ータあり
                        else:
                            before_L_stec = L_stec[isat, iepoc-ibefore]
                            # サイクルスリ�?�? -> start
                            if abs(now_L_stec-before_L_stec) > threshold:
                                if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                    L_info[isat][iepoc] = 4
                            # 非サイクルスリ�?�? -> continue
                            else:
                                if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                    L_info[isat][iepoc] = 3
                    # ep=1のL�?ータな�?
                    else:
                        L_info[isat][iepoc] = 0

        # for iepoch in range(2880):
        #     print(L_info[26, iepoch], L_stec[26, iepoch])

        for isat in range(Gmaxsat):
            # 直前�?�L_info=2
            before_slip = 0
            count = 0
            for iepoc in range(2880):
                # �?ータな�?
                # �?ータ数は増やさな�?
                if L_info[isat][iepoc] == 0:
                    pass
                # �?ータあり、スタートでな�?
                # �?ータ数�?け増や�?
                elif L_info[isat][iepoc] == 1:
                    count += 1
                # �?ータあり、スター�?
                elif L_info[isat][iepoc] == 2 or L_info[isat][iepoc] == 4:
                    # 直前�?�L_info=2からの�?ータ数が規定値以�? -> 有効
                    if count >= Valid_Data_Length:
                        for jepoc in range(before_slip, iepoc):
                            if L_info[isat][jepoc] == 1 or L_info[isat][jepoc] == 2:
                                L_valid[isat][jepoc] = 2
                            elif L_info[isat][jepoc] == 3 or L_info[isat][jepoc] == 4:
                                L_valid[isat][jepoc] = 1
                    before_slip = iepoc
                    count = 0
                if iepoc == 2879:
                    if count >= Valid_Data_Length:
                        for jepoc in range(before_slip, 2880):
                            if L_info[isat][jepoc] == 1 or L_info[isat][jepoc] == 2:
                                L_valid[isat][jepoc] = 2
                            elif L_info[isat][jepoc] == 3 or L_info[isat][jepoc] == 4:
                                L_valid[isat][jepoc] = 1
        for isat in range(Gmaxsat):
            for iepoc in range(2880):
                # �?ータな�?
                if L_info[isat][iepoc] == 0:
                    pass
                # スタートでな�?
                elif L_info[isat][iepoc] == 1 or L_info[isat][iepoc] == 3:
                    if L_valid[isat][iepoc]:
                        L_data_set[isat][iepoc] = n_data
                        L_data_len[n_data] += 1
                # スター�?
                elif L_info[isat][iepoc] == 2 or L_info[isat][iepoc] == 4:
                    # 有効
                    if L_valid[isat][iepoc]:
                        n_data += 1
                        L_data_set[isat][iepoc] = n_data
                        L_data_sat_idx[n_data] = isat
                        L_data_begin[n_data] = iepoc
                        L_data_len[n_data] += 1

        n_data += 1

        dt3 = time.time()-dt_now

        P_O = XYZ()
        P_O.set(p_o[0], p_o[1], p_o[2])
        # Determine integer ambiguity B for data set {idata}
        B_data = np.full((n_data), 0.0, dtype=float)
        # print(E_orbit_data[307,9])
        for idata in range(n_data):
            isat_idx = L_data_sat_idx[idata]
            iblock_bgn = L_data_begin[idata]
            # 最後�?�ブロ�?ク
            if idata == n_data-1:
                nextblock_bgn = 2880
            # 同じ衛星の次のブロ�?クのはじまりまで
            elif isat_idx == L_data_sat_idx[idata+1]:
                nextblock_bgn = L_data_begin[idata+1]
            # 別の衛星
            else:
                nextblock_bgn = 2880
            # print(idata, "/", n_data, sat_list[isat_idx])
            lower = 0.0
            upper = 0.0
            if "G" in sat_list[isat_idx]:
                isat = int(sat_list[isat_idx][1:3])
                for iepoc in range(iblock_bgn, nextblock_bgn):
                    # �?ータな�?
                    if L_info[isat, iepoc] == 0:
                        pass
                    elif L_info[isat, iepoc] == 1 or L_info[isat, iepoc] == 2:
                        if L_valid[isat, iepoc]:
                            jORBIT = XYZ()
                            jORBIT.set(G_orbit_data[iepoc, isat, 0],
                                       G_orbit_data[iepoc, isat, 1], G_orbit_data[iepoc, isat, 2])
                            lL = L_stec[isat, iepoc]
                            lP = P_stec[isat, iepoc]
                            j_ipp = specify_H(P_O, jORBIT, H_ipp*1.0e+3)
                            lZ = zenith(P_O, j_ipp)
                            lower += np.power(np.sin(lZ), 2)
                            upper += (lP-lL)*np.power(np.sin(lZ), 2)
                    else:
                        pass
                B = upper/lower
                B_data[idata] = B

        # print(n_data)
        # for idata in range(n_data):
        #     print(idata,":",sat_list[L_data_sat_idx[idata]],L_data_begin[idata],L_data_len[idata],B_data[idata])
        # input()

        dt4 = time.time()-dt_now

        mod_L_stec = np.full((maxsat, 2880), 0.0, dtype=float)

        for idata in range(n_data):
            isat = L_data_sat_idx[idata]
            iblock_bgn = L_data_begin[idata]
            # 最後�?�ブロ�?ク
            if idata == n_data-1:
                nextblock_bgn = 2880
            # 同じ衛星の次のブロ�?クのはじまりまで
            elif isat == L_data_sat_idx[idata+1]:
                nextblock_bgn = L_data_begin[idata+1]
            # 別の衛星
            else:
                nextblock_bgn = 2880
            iB = B_data[idata]
            # print(idata,iblock_bgn,nextblock_bgn,iB)
            for iepoc in range(iblock_bgn, nextblock_bgn):
                # �?ータな�?
                if L_info[isat, iepoc] == 0:
                    pass
                elif L_info[isat, iepoc] > 0:
                    mod_L_stec[isat, iepoc] = L_stec[isat, iepoc]+iB
                else:
                    pass
        # for iepoc in range(2880):
        #     print(iepoc,":",L_info[30,iepoc],L_stec[30,iepoc],P_stec[30,iepoc],mod_L_stec[30,iepoc],Zs[30,iepoc])
        # input()

        for isat in range(len(sat_list)):
            if np.any(L_stec_bool[isat, :]):
                path_nw = "{md}/{c}/{y:04}/{d:03d}/{sat}".format(
                    md=mdf_folder, y=year4, d=day, sat=sat_list[isat], c=country)
                # print(isat,sat_list[isat])
                os.makedirs(path_nw, exist_ok=True)
                path_nd = path_nw + "/{s:4}.mdf".format(s=sta)
                sat_num = int(sat_list[isat][1:3])
                if "G" in sat_list[isat]:
                    f_l1 = f_l1_GPS
                    f_l2 = f_l2_GPS
                    # print(sat_list[isat],sat_num,G_orbit_bool[:,sat_num])
                    # isat ... index of L_stec
                    # sat_num ... number id of satelite
                    make_tomo_file(path_nd, L_valid, mod_L_stec, P_stec, L_stec, L_stec_bool, P_stec_bool,
                                   G_orbit_data, G_orbit_bool, p_o, isat, sat_num, f_l1, f_l2, sat_list[isat])

        print("{y:04} {d:03} {s} end : {t:09.3f} ( {d1:09.3f} {d2:09.3f} {d3:09.3f} {d4:09.3f} )".format(
            y=year4, d=day, s=sta, t=time.time()-dt_now, d1=dt1, d2=dt2, d3=dt3, d4=dt4))


def rnx2mdf_v2_12(year4: int, day: int, stations: list):
    year2 = year4 % 100
    md = datetime.date(year4, 1, 1)+datetime.timedelta(day-1)

    obs_day = datetime.datetime(year4, md.month, md.day)

    G_orbit_data = np.full((2880, Gmaxsat, 3), 0.0, dtype=float)
    G_orbit_bool = np.full((2880, Gmaxsat), False, dtype=bool)
    G_finish_check = np.full((Gmaxsat), False, dtype=bool)
    E_orbit_data = np.full((2880, Emaxsat, 3), 0.0, dtype=float)
    E_orbit_bool = np.full((2880, Emaxsat), False, dtype=bool)
    E_finish_check = np.full((Emaxsat), False, dtype=bool)
    J_orbit_data = np.full((2880, Jmaxsat, 3), 0.0, dtype=float)
    J_orbit_bool = np.full((2880, Jmaxsat), False, dtype=bool)
    J_finish_check = np.full((Jmaxsat), False, dtype=bool)

    for sta in stations:
        # print(E_orbit_data[307,9])
        # About 15 sec for each station (ver 1.0)
        # About 7 sec for each station (ver 1.2)
        # About 15 sec for each station (ver 2.0)
        # Read observation data
        path_o = path_tec + \
            "{s}{d:03d}0.{y2:02}o".format(y=year4, y2=year2, d=day, s=sta)
        # L_stec,L_stec_bool,P_stec,P_stec_bool
        # [i][j] ... index i(sat_dict[i]) satelite, epoch j data
        # sat_bool[i] ... id i satelite
        # sat_dict[id] ... L_stec index of sat id
        # sat_list[i] ... index i satelite
        sat_dict, sat_list, L_stec, L_stec_bool, P_stec, P_stec_bool, p_o, obs_day = load_observ_file_v2_12(
            path_o)
        # print(sat_list)

        dt1 = time.time()-dt_now

        if not np.all(G_finish_check):
            GPS_file = path_tec + \
                "{s}{d:03}0.{y2:02}n".format(s=sta, d=day, y2=year2)

            if os.path.exists(GPS_file):
                # idsat[k] ... k th record satelite id
                # oel[k][0-27] ... k th record
                # tiempo[k] ... k th record time from obs_day 00:00:00UTC
                # i1stsat[k] ... list of k th satelite record number
                G_idsat, G_tiempo, G_oel, G_i1stsat = load_navi_file_v2(
                    GPS_file, obs_day)

            for isat in range(1, Gmaxsat):
                # if dont finish
                if not G_finish_check[isat]:
                    Gor_data = make_orbit_data(
                        isat, G_tiempo, G_oel, G_i1stsat)
                    for iepoc in range(2880):
                        if sum(abs(Gor_data[iepoc, :])) > 200.0:
                            G_orbit_data[iepoc, isat,
                                         :] = Gor_data[iepoc, :]
                            G_orbit_bool[iepoc, isat] = True

                if np.all(G_orbit_bool[:, isat]):
                    G_finish_check[isat] = True

        if not np.all(E_finish_check):
            Galileo_file = path_tec + \
                "{s}{d:03}0.{y2:02}l".format(s=sta, d=day, y2=year2)

            if os.path.exists(Galileo_file):
                # idsat[k] ... k th record satelite id
                # oel[k][0-27] ... k th record
                # tiempo[k] ... k th record time from obs_day 00:00:00UTC
                # i1stsat[k] ... list of k th satelite record number
                E_idsat, E_tiempo, E_oel, E_i1stsat = load_navi_file(
                    Galileo_file, obs_day)

            for isat in range(1, Emaxsat):
                if not E_finish_check[isat]:
                    Eor_data = make_orbit_data(
                        isat, E_tiempo, E_oel, E_i1stsat)
                    for iepoc in range(2880):
                        if sum(abs(Eor_data[iepoc, :])) > 200.0:
                            E_orbit_data[iepoc, isat,
                                         :] = Eor_data[iepoc, :]
                            E_orbit_bool[iepoc, isat] = True

                if np.all(E_orbit_bool[:, isat]):
                    E_finish_check[isat] = True

        if not np.all(J_finish_check):
            QZSS_file = path_tec + \
                "{s}{d:03}0.{y2:02}q".format(s=sta, d=day, y2=year2)

            if os.path.exists(QZSS_file):
                # idsat[k] ... k th record satelite id
                # oel[k][0-27] ... k th record
                # tiempo[k] ... k th record time from obs_day 00:00:00UTC
                # i1stsat[k] ... list of k th satelite record number
                J_idsat, J_tiempo, J_oel, J_i1stsat = load_navi_file_v2(
                    QZSS_file, obs_day, code=1)

            for isat in range(1, Jmaxsat):
                if not J_finish_check[isat]:
                    Jor_data = make_orbit_data(
                        isat, J_tiempo, J_oel, J_i1stsat)
                    for iepoc in range(2880):
                        if sum(abs(Jor_data[iepoc, :])) > 200.0:
                            J_orbit_data[iepoc, isat,
                                         :] = Jor_data[iepoc, :]
                            J_orbit_bool[iepoc, isat] = True

                if np.all(J_orbit_bool[:, isat]):
                    J_finish_check[isat] = True

        dt2 = time.time()-dt_now

        # L_data_set[i,j]...number of set, sat i, epoch j
        # L_data_len[i] ... length of set i
        # L_data_sat[i] ... satelite of set i
        # L_data_begin[i] ... first epoch of set i
        L_data_set = np.full((maxsat, 2880), -1, dtype=int)
        L_data_len = np.full((2000), 0, dtype=int)
        L_data_sat_idx = np.full((2000), 0, dtype=int)
        L_data_begin = np.full((2000), 0, dtype=int)
        # 0 ... �?ータな�?
        # 1 ... L,P, not start
        # 2 ... L,P, start
        # 3 ... L, not start
        # 4 ... L, start
        L_info = np.full((maxsat, 2880), 0, dtype=int)
        # 0 ... �?ータ無効
        # 1 ... �?ータ有効 but 計算には�?れな�?
        # 2 ... �?ータ有効 and 計算に�?れる
        L_valid = np.full((maxsat, 2880), 0, dtype=int)
        Zs = np.full((maxsat, 2880), 0.0, dtype=float)
        n_data = -1

        for isat in range(len(sat_list)):
            sat_id = sat_list[isat]
            # print(sat_id)
            rec = XYZ()
            rec.set(p_o[0], p_o[1], p_o[2])
            iZ = 90.0
            for iepoc in range(2880):
                if "G" in sat_id:
                    if not G_orbit_bool[iepoc, int(sat_id[1:3])]:
                        L_info[isat, iepoc] = 0
                        continue
                    sat_x = G_orbit_data[iepoc, int(sat_id[1:3]), 0]
                    sat_y = G_orbit_data[iepoc, int(sat_id[1:3]), 1]
                    sat_z = G_orbit_data[iepoc, int(sat_id[1:3]), 2]
                    sat = XYZ()
                    sat.set(sat_x, sat_y, sat_z)
                    iZ = r_zenith(rec, sat)
                elif "E" in sat_id:
                    if not E_orbit_bool[iepoc, int(sat_id[1:3])]:
                        L_info[isat, iepoc] = 0
                        continue
                    sat_x = E_orbit_data[iepoc, int(sat_id[1:3]), 0]
                    sat_y = E_orbit_data[iepoc, int(sat_id[1:3]), 1]
                    sat_z = E_orbit_data[iepoc, int(sat_id[1:3]), 2]
                    sat = XYZ()
                    sat.set(sat_x, sat_y, sat_z)
                    iZ = r_zenith(rec, sat)
                elif "J" in sat_id:
                    if not J_orbit_bool[iepoc, int(sat_id[1:3])]:
                        L_info[isat, iepoc] = 0
                        continue
                    sat_x = J_orbit_data[iepoc, int(sat_id[1:3]), 0]
                    sat_y = J_orbit_data[iepoc, int(sat_id[1:3]), 1]
                    sat_z = J_orbit_data[iepoc, int(sat_id[1:3]), 2]
                    sat = XYZ()
                    sat.set(sat_x, sat_y, sat_z)
                    iZ = r_zenith(rec, sat)
                Zs[isat][iepoc] = iZ*180./math.pi
                if iepoc == 0:
                    # L,P両方�?ータがあ�? -> 新しいブロ�?クの始ま�?
                    if L_stec_bool[isat][iepoc] and P_stec_bool[isat][iepoc]:
                        if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                            L_info[isat][iepoc] = 2
                    # Lのみある -> 4
                    elif L_stec_bool[isat][iepoc]:
                        if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                            L_info[isat][iepoc] = 4
                    # どちらも初めの�?ータがな�? -> �?ータな�?
                    else:
                        L_info[isat][iepoc] = 0
                else:
                    now_L_stec = L_stec[isat, iepoc]
                    # ep=iepocのP,Lデータあり
                    if L_stec_bool[isat][iepoc] and P_stec_bool[isat][iepoc]:
                        ibefore = -1
                        for jbefore in range(1, min(null_threshold+2, iepoc+1)):
                            if L_info[isat][iepoc-jbefore] > 0:
                                ibefore = jbefore
                                break
                        # 前に有効なデータなし -> start
                        if ibefore == -1:
                            if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                L_info[isat][iepoc] = 2
                        # 有効なデータあり
                        else:
                            before_L_stec = L_stec[isat, iepoc-ibefore]
                            # サイクルスリップ -> start
                            if abs(now_L_stec-before_L_stec) > threshold:
                                if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                    L_info[isat][iepoc] = 2
                            # 非サイクルスリップ -> continue
                            else:
                                if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                    L_info[isat][iepoc] = 1
                    # ep=iepocのLデータのみ
                    elif L_stec_bool[isat][iepoc]:
                        ibefore = -1
                        for jbefore in range(1, min(null_threshold+2, iepoc)):
                            if L_info[isat][iepoc-jbefore] > 0:
                                ibefore = jbefore
                        # 前に有効なデータなし -> start
                        if ibefore == -1:
                            if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                L_info[isat][iepoc] = 4
                        # 有効なデータあり
                        else:
                            before_L_stec = L_stec[isat, iepoc-ibefore]
                            # サイクルスリ�?�? -> start
                            if abs(now_L_stec-before_L_stec) > threshold:
                                if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                    L_info[isat][iepoc] = 4
                            # 非サイクルスリ�?�? -> continue
                            else:
                                if zenith_threshold-90.0 < iZ*180./math.pi < 90.0-zenith_threshold:
                                    L_info[isat][iepoc] = 3
                    # ep=1のL�?ータな�?
                    else:
                        L_info[isat][iepoc] = 0

        # for isat in range(0, 1):
        #     for iepoc in range(2880):
        #         if L_info[isat, iepoc] == 0:
        #             print(sat_list[isat], iepoc, ":", L_stec_bool[isat, iepoc], P_stec_bool[isat, iepoc], L_stec[isat,
        #                                                                                                          iepoc], P_stec[isat, iepoc], L_info[isat, iepoc], L_valid[isat, iepoc], Zs[isat, iepoc])
        #         if L_info[isat, iepoc] == 1 or L_info[isat, iepoc] == 3:
        #             print("*", sat_list[isat], iepoc, ":", L_stec_bool[isat, iepoc], P_stec_bool[isat, iepoc], L_stec[isat,
        #                                                                                                               iepoc], P_stec[isat, iepoc], L_info[isat, iepoc], L_valid[isat, iepoc], Zs[isat, iepoc])
        #         if L_info[isat, iepoc] == 2 or L_info[isat, iepoc] == 4:
        #             print("***", sat_list[isat], iepoc, ":", L_stec_bool[isat, iepoc], P_stec_bool[isat, iepoc], L_stec[isat,
        #                                                                                                                 iepoc], P_stec[isat, iepoc], L_info[isat, iepoc], L_valid[isat, iepoc], Zs[isat, iepoc], "***")
        # #     input()

        for isat in range(maxsat):
            # 直前�?�L_info=2
            before_slip = 0
            count = 0
            for iepoc in range(2880):
                # �?ータな�?
                # �?ータ数は増やさな�?
                if L_info[isat][iepoc] == 0:
                    pass
                # �?ータあり、スタートでな�?
                # �?ータ数�?け増や�?
                elif L_info[isat][iepoc] == 1:
                    count += 1
                # �?ータあり、スター�?
                elif L_info[isat][iepoc] == 2 or L_info[isat][iepoc] == 4:
                    # 直前�?�L_info=2からの�?ータ数が規定値以�? -> 有効
                    if count >= Valid_Data_Length:
                        for jepoc in range(before_slip, iepoc):
                            if L_info[isat][jepoc] == 1 or L_info[isat][jepoc] == 2:
                                L_valid[isat][jepoc] = 2
                            elif L_info[isat][jepoc] == 3 or L_info[isat][jepoc] == 4:
                                L_valid[isat][jepoc] = 1
                    before_slip = iepoc
                    count = 0
                if iepoc == 2879:
                    if count >= Valid_Data_Length:
                        for jepoc in range(before_slip, 2880):
                            if L_info[isat][jepoc] == 1 or L_info[isat][jepoc] == 2:
                                L_valid[isat][jepoc] = 2
                            elif L_info[isat][jepoc] == 3 or L_info[isat][jepoc] == 4:
                                L_valid[isat][jepoc] = 1
        for isat in range(maxsat):
            for iepoc in range(2880):
                # �?ータな�?
                if L_info[isat][iepoc] == 0:
                    pass
                # スタートでな�?
                elif L_info[isat][iepoc] == 1 or L_info[isat][iepoc] == 3:
                    if L_valid[isat][iepoc] > 0:
                        L_data_set[isat][iepoc] = n_data
                        L_data_len[n_data] += 1
                # スター�?
                elif L_info[isat][iepoc] == 2 or L_info[isat][iepoc] == 4:
                    # 有効
                    if L_valid[isat][iepoc] > 0:
                        n_data += 1
                        L_data_set[isat][iepoc] = n_data
                        L_data_sat_idx[n_data] = isat
                        L_data_begin[n_data] = iepoc
                        L_data_len[n_data] += 1

        n_data += 1
        dt3 = time.time()-dt_now
        # print(L_data_sat)
        P_O = XYZ()
        P_O.set(p_o[0], p_o[1], p_o[2])
        B_data = np.full((n_data), 0.0, dtype=float)
        for idata in range(n_data):
            # print(idata, "/", n_data)
            isat_idx = L_data_sat_idx[idata]
            iblock_bgn = L_data_begin[idata]
            # 最後�?�ブロ�?ク
            if idata == n_data-1:
                nextblock_bgn = 2880
            # 同じ衛星の次のブロ�?クのはじまりまで
            elif isat_idx == L_data_sat_idx[idata+1]:
                nextblock_bgn = L_data_begin[idata+1]
            # 別の衛星
            else:
                nextblock_bgn = 2880
            # print(idata, "/", n_data, sat_list[isat_idx])
            lower = 0.0
            upper = 0.0
            if "G" in sat_list[isat_idx]:
                isat = int(sat_list[isat_idx][1:3])
                for iepoc in range(iblock_bgn, nextblock_bgn):
                    # �?ータな�?
                    if L_info[isat_idx][iepoc] == 0:
                        pass
                    elif L_info[isat_idx][iepoc] == 1 or L_info[isat_idx][iepoc] == 2:
                        if L_valid[isat_idx][iepoc]:
                            jORBIT = XYZ()
                            jORBIT.set(G_orbit_data[iepoc, isat, 0],
                                       G_orbit_data[iepoc, isat, 1], G_orbit_data[iepoc, isat, 2])
                            lL = L_stec[isat_idx, iepoc]
                            lP = P_stec[isat_idx, iepoc]
                            j_ipp = specify_H(P_O, jORBIT, H_ipp*1.0e+3)
                            lZ = zenith(P_O, j_ipp)
                            lower += np.power(np.sin(lZ), 2)
                            upper += (lP-lL)*np.power(np.sin(lZ), 2)
                    else:
                        pass
                B = upper/lower
                B_data[idata] = B

            elif "E" in sat_list[isat_idx]:
                isat = int(sat_list[isat_idx][1:3])
                for iepoc in range(iblock_bgn, nextblock_bgn):
                    # �?ータな�?
                    if L_info[isat_idx][iepoc] == 0:
                        pass
                    elif L_info[isat_idx][iepoc] == 1 or L_info[isat_idx][iepoc] == 2:
                        if L_valid[isat_idx][iepoc]:
                            jORBIT = XYZ()
                            jORBIT.set(E_orbit_data[iepoc, isat, 0],
                                       E_orbit_data[iepoc, isat, 1], E_orbit_data[iepoc, isat, 2])
                            lL = L_stec[isat_idx, iepoc]
                            lP = P_stec[isat_idx, iepoc]
                            j_ipp = specify_H(P_O, jORBIT, H_ipp*1.0e+3)
                            lZ = zenith(P_O, j_ipp)
                            lower += np.power(np.sin(lZ), 2)
                            upper += (lP-lL)*np.power(np.sin(lZ), 2)
                    else:
                        pass
                B = upper/lower
                B_data[idata] = B
            elif "J" in sat_list[isat_idx]:
                isat = int(sat_list[isat_idx][1:3])
                for iepoc in range(iblock_bgn, nextblock_bgn):
                    # �?ータな�?
                    if L_info[isat_idx][iepoc] == 0:
                        pass
                    elif L_info[isat_idx][iepoc] == 1 or L_info[isat_idx][iepoc] == 2:
                        if L_valid[isat_idx][iepoc]:
                            jORBIT = XYZ()
                            jORBIT.set(J_orbit_data[iepoc, isat, 0],
                                       J_orbit_data[iepoc, isat, 1], J_orbit_data[iepoc, isat, 2])
                            lL = L_stec[isat_idx, iepoc]
                            lP = P_stec[isat_idx, iepoc]
                            j_ipp = specify_H(P_O, jORBIT, H_ipp*1.0e+3)
                            lZ = zenith(P_O, j_ipp)
                            lower += np.power(np.sin(lZ), 2)
                            upper += (lP-lL)*np.power(np.sin(lZ), 2)
                    else:
                        pass
                B = upper/lower
                B_data[idata] = B

        # for isat in range(0, 1):
        #     for iepoc in range(2880):
        #         if L_info[isat, iepoc] == 0:
        #             print(sat_list[isat], iepoc, ":", L_stec_bool[isat, iepoc], P_stec_bool[isat, iepoc], L_stec[isat,
        #                                                                                                          iepoc], P_stec[isat, iepoc], L_info[isat, iepoc], L_valid[isat, iepoc], Zs[isat, iepoc])
        #         if L_info[isat, iepoc] == 1 or L_info[isat, iepoc] == 3:
        #             print("*", sat_list[isat], iepoc, ":", L_stec_bool[isat, iepoc], P_stec_bool[isat, iepoc], L_stec[isat,
        #                                                                                                               iepoc], P_stec[isat, iepoc], L_info[isat, iepoc], L_valid[isat, iepoc], Zs[isat, iepoc])
        #         if L_info[isat, iepoc] == 2 or L_info[isat, iepoc] == 4:
        #             print("***", sat_list[isat], iepoc, ":", L_stec_bool[isat, iepoc], P_stec_bool[isat, iepoc], L_stec[isat,
        #                                                                                                                 iepoc], P_stec[isat, iepoc], L_info[isat, iepoc], L_valid[isat, iepoc], Zs[isat, iepoc], "***")
        #     input()
        # for idata in range(n_data):
        #     print(idata, sat_list[L_data_sat_idx[idata]], L_data_begin[idata])
        # input()
        dt4 = time.time()-dt_now
        # print(dt4)
        # mod_L_stec[i][j] ... satelite i, epoch j
        mod_L_stec = np.full((maxsat, 2880), 0.0, dtype=float)

        for idata in range(n_data):
            isat = L_data_sat_idx[idata]
            iblock_bgn = L_data_begin[idata]
            # 最後�?�ブロ�?ク
            if idata == n_data-1:
                nextblock_bgn = 2880
            # 同じ衛星の次のブロ�?クのはじまりまで
            elif isat == L_data_sat_idx[idata+1]:
                nextblock_bgn = L_data_begin[idata+1]
            # 別の衛星
            else:
                nextblock_bgn = 2880
            # print(idata, ":", sat_list[L_data_sat_idx[idata]],
            #       L_data_begin[idata], nextblock_bgn, L_data_len[idata], B_data[idata], isat, "->", L_data_sat_idx[idata+1])
            iB = B_data[idata]
            for iepoc in range(iblock_bgn, nextblock_bgn):
                # �?ータな�?
                if L_info[isat][iepoc] == 0:
                    pass
                else:
                    if L_valid[isat][iepoc] > 0:
                        mod_L_stec[isat, iepoc] = L_stec[isat, iepoc]+iB

        for isat in range(len(sat_list)):
            path_nw = "{md}/{c}/{y:04}/{d:03d}/{sat}".format(
                md=mdf_folder, y=year4, d=day, sat=sat_list[isat], c=country)
            path_nd = path_nw + "/{s:4}.mdf".format(s=sta)
            sat_num = int(sat_list[isat][1:3])
            if "G" in sat_list[isat]:
                os.makedirs(path_nw, exist_ok=True)
                f_l1 = f_lA_GPS
                f_l2 = f_l2_GPS
                # print(sat_list[isat],sat_num,G_orbit_bool[:,sat_num])
                # isat ... index of L_stec
                # sat_num ... number id of satelite
                make_tomo_file(path_nd, L_valid, mod_L_stec, P_stec, L_stec, L_stec_bool, P_stec_bool,
                               G_orbit_data, G_orbit_bool, p_o, isat, sat_num, f_l1, f_l2, sat_list[isat])
            elif "E" in sat_list[isat]:
                os.makedirs(path_nw, exist_ok=True)
                f_l1 = f_l1_Galileo
                f_l2 = f_l5_Galileo
                make_tomo_file(path_nd, L_valid, mod_L_stec, P_stec, L_stec, L_stec_bool, P_stec_bool,
                               E_orbit_data, E_orbit_bool, p_o, isat, sat_num, f_l1, f_l2, sat_list[isat])
            elif "J" in sat_list[isat]:
                os.makedirs(path_nw, exist_ok=True)
                f_l1 = f_lC_QZSS
                f_l2 = f_l5_QZSS
                make_tomo_file(path_nd, L_valid, mod_L_stec, P_stec, L_stec, L_stec_bool, P_stec_bool,
                               J_orbit_data, J_orbit_bool, p_o, isat, sat_num, f_l1, f_l2, sat_list[isat])

        print("{y:04} {d:03} {s} end : {t:09.3f} ( {d1:09.3f} {d2:09.3f} {d3:09.3f} {d4:09.3f} )".format(
            y=year4, d=day, s=sta, t=time.time()-dt_now, d1=dt1, d2=dt2, d3=dt3, d4=dt4))


def rnx2mdf(year4: int, day: int, stations: list, version: str):
    if version == "v2_10":
        return rnx2mdf_v2_10(year4, day, stations)
    if version == "v3_02":
        return rnx2mdf_v3_02(year4, day, stations)
    if version == "v2_12":
        return rnx2mdf_v2_12(year4, day, stations)


def mdf2bias_v3_02(year4: int, day: int, smooth: bool):
    start = time.time()
    # station_dict[受信局ID]=割り振られた番号
    station_dict = {}
    satelite_list = []
    n_rec = 0
    for isat in range(Gmaxsat):
        modified_TEC = "{md}/{c}/{y:04}/{d:03d}/G{sat:02}".format(
            md=mdf_folder, y=year4, d=day, sat=isat, c=country)
        if os.path.isdir(modified_TEC):
            # print(modified_TEC)
            satelite_list.append("G{sat:02}".format(sat=isat))
            mod_files = glob.glob(modified_TEC+"/*.mdf")
            for mod_file in mod_files:
                # print(mod_file)
                receiver_code = mod_file[-8:-4]
                if not receiver_code in station_dict:
                    station_dict[receiver_code] = n_rec
                    n_rec += 1
    for isat in range(Emaxsat):
        modified_TEC = "{md}/{c}/{y:04}/{d:03d}/E{sat:02}".format(
            md=mdf_folder, y=year4, d=day, sat=isat, c=country)
        if os.path.isdir(modified_TEC):
            # print(modified_TEC)
            satelite_list.append("E{sat:02}".format(sat=isat))
            mod_files = glob.glob(modified_TEC+"/*.mdf")
            for mod_file in mod_files:
                # print(mod_file)
                receiver_code = mod_file[-8:-4]
                if not receiver_code in station_dict:
                    station_dict[receiver_code] = n_rec
                    n_rec += 1
    for isat in range(Jmaxsat):
        modified_TEC = "{md}/{c}/{y:04}/{d:03d}/J{sat:02}".format(
            md=mdf_folder, y=year4, d=day, sat=isat, c=country)
        if os.path.isdir(modified_TEC):
            # print(modified_TEC)
            satelite_list.append("J{sat:02}".format(sat=isat))
            mod_files = glob.glob(modified_TEC+"/*.mdf")
            for mod_file in mod_files:
                # print(mod_file)
                receiver_code = mod_file[-8:-4]
                if not receiver_code in station_dict:
                    station_dict[receiver_code] = n_rec
                    n_rec += 1
    # for isat in range(Rmaxsat):
    #     modified_TEC = "E:/modified_data/{c}/{y:04}/{d:03d}/R{sat:02}".format(
    #         y=year4, d=day, sat=isat, c=country)
    #     if os.path.isdir(modified_TEC):
    #         # print(modified_TEC)
    #         satelite_list.append("R{sat:02}".format(sat=isat))
    #         mod_files = glob.glob(modified_TEC+"/*.mdf")
    #         for mod_file in mod_files:
    #             # print(mod_file)
    #             receiver_code = mod_file[-8:-4]
    #             if not receiver_code in station_dict:
    #                 station_dict[receiver_code] = n_rec
    #                 n_rec += 1
    n_record = np.full((len(satelite_list)), 0, dtype=int)
    # station_dict_swap[割り振られた番号]=受信局ID
    station_dict_swap = {v: k for k, v in station_dict.items()}
    # Region where TEC value is assumed to be constant
    TEC_ident_epoch = 30  # [epoch]  0.1[hour] = 6[min] = 12[epoch]
    TEC_ident_area = 1.0  # deg
    # Time where satelite and receiver bias to be constant
    bias_ident_time = 2880  # [epoch] 2880[epoch] = 1440[min] = 24[hour]
    EPOCH_PAR_DAY = 2880
    HOUR_PAR_EPOCH = 24.0/2880

    m_LAT = 15.0
    M_LAT = 65.0
    m_LON = 115.0
    M_LON = 165.0

    n_LAT = round((M_LAT-m_LAT)/TEC_ident_area)
    n_LON = round((M_LON-m_LON)/TEC_ident_area)

    I = math.ceil(EPOCH_PAR_DAY/TEC_ident_epoch) * \
        n_LAT*n_LON     # Number of pixel
    J = len(satelite_list)              # Number of satelite
    nbias = 3
    # GPS ... 1
    # Galileo,QZSS ... 1
    K = len(station_dict)*nbias   # (Number of receiver) X (Number of DCB)
    nrec = len(station_dict)

    Mdata = math.ceil(EPOCH_PAR_DAY/TEC_ident_epoch)*K*20
    print("Set VTEC mapping situation. :", I, J, nrec, K, Mdata,
          "{t:.3f}".format(t=time.time()-dt_now))  # 0.121[s]

    raw_stec_data = np.full((Mdata), 0.0, dtype=float)
    raw_sat_data = np.full((Mdata, 3), 0.0, dtype=float)
    raw_rec_data = np.full((Mdata, 3), 0.0, dtype=float)
    rec_num = np.full((Mdata), 0, dtype=int)
    sat_num = np.full((Mdata), 0, dtype=int)
    time_num = np.full((Mdata), 0, dtype=int)
    sat_idxs = np.full((Mdata), 0, dtype=int)
    idx = 0
    Epochlist = []
    for i in range(round(EPOCH_PAR_DAY/TEC_ident_epoch)):
        Epochlist.append(TEC_ident_epoch*i)
    print(Epochlist)
    n_TIME = len(Epochlist)
    for isat in range(len(satelite_list)):
        modified_TEC = "{md}/{c}/{y:04}/{d:03d}/{sat}".format(
            md=mdf_folder, y=year4, d=day, sat=satelite_list[isat], c=country)
        satid = satelite_list[isat]
        if os.path.isdir(modified_TEC):
            mod_files = glob.glob(modified_TEC+"\*.mdf")
            for mod_file in mod_files:
                # print(mod_file)
                with open(mod_file, "r") as m_f:
                    recid = mod_file[-8:-4]
                    # Header
                    while True:
                        line = m_f.readline()
                        # print(line)
                        if "END OF HEADER" in line:
                            break
                        elif "#" in line:
                            continue
                        elif not line:
                            break
                    # Data
                    Timeidx = 0
                    while True:
                        line = m_f.readline()
                        if not line:
                            break
                        #  print(line)
                        dline = line.split()
                        # print(dline)
                        epoch = round(float(dline[0])*120.0)
                        if epoch in Epochlist:
                            n_record[isat] += 1
                            rawstec = float(dline[1])
                            satx = float(dline[4])
                            saty = float(dline[5])
                            satz = float(dline[6])
                            recx = float(dline[7])
                            recy = float(dline[8])
                            recz = float(dline[9])
                            raw_stec_data[idx] = rawstec
                            raw_sat_data[idx, 0] = satx
                            raw_sat_data[idx, 1] = saty
                            raw_sat_data[idx, 2] = satz
                            raw_rec_data[idx, 0] = recx
                            raw_rec_data[idx, 1] = recy
                            raw_rec_data[idx, 2] = recz
                            rec_num[idx] = station_dict[recid]
                            sat_num[idx] = isat
                            time_num[idx] = Epochlist.index(epoch)
                            if "G" in satid:
                                sat_idxs[idx] = 0
                            elif "E" in satid:
                                sat_idxs[idx] = 1
                            elif "J" in satid:
                                sat_idxs[idx] = 1
                            idx += 1

    # for i in range(idx):
    #     print(i,":",raw_stec_data[i],raw_sat_data[i],raw_rec_data[i],rec_num[i],sat_num[i],time_num[i])
    # A=input()
    print("Number of Record", n_record)
    ndata = idx-1
    print("End importing raw stec data. Data number :", ndata,
          "{t:.3f}".format(t=time.time()-dt_now))  # 52.859[s]
    left_lil = lil_matrix((ndata, I+J+K))
    right = [0.0 for i in range(ndata)]

    for i in range(ndata):
        # G ... 0 , Q,E ... 1
        iband = sat_idxs[i]
        isat = sat_num[i]
        irec = rec_num[i]
        itime = time_num[i]
        iSTEC = raw_stec_data[i]
        iREC = XYZ()
        iREC.set(raw_rec_data[i, 0],
                 raw_rec_data[i, 1], raw_rec_data[i, 2])
        iSAT = XYZ()
        iSAT.set(raw_sat_data[i, 0],
                 raw_sat_data[i, 1], raw_sat_data[i, 2])
        iIPP = specify_H(iREC, iSAT, H_ipp*1.0e+3)
        i1IPP = specify_H(iREC, iSAT, H1_ipp*1.0e+3)
        i2IPP = specify_H(iREC, iSAT, H2_ipp*1.0e+3)
        S1 = (i1IPP-i2IPP).L2()
        S0 = (H2_ipp-H1_ipp)*1.0e+3
        iIPP_blh = iIPP.to_BLH()
        # print(i,":",isat,irec,itime,iSTEC,iIPP_blh)
        iIPP_b = iIPP_blh.b
        iIPP_l = iIPP_blh.l
        iIPP_b_idx = math.floor((iIPP_b-m_LAT)/TEC_ident_area)
        iIPP_l_idx = math.floor((iIPP_l-m_LON)/TEC_ident_area)
        if -1 < iIPP_b_idx < n_LAT and -1 < iIPP_l_idx < n_LON:
            iTEC_idx = iIPP_b_idx*n_LON+iIPP_l_idx
            # for VTEC
            left_lil[i, iTEC_idx+itime*n_LAT*n_LON] = S1/S0
            # for satelite bias
            left_lil[i, I+isat] = 1.0
            # for receiver bias
            left_lil[i, I+J+iband*nrec+irec] = 1.0
            # for right side
            right[i] = (iSTEC)
            # print(iIDX)
        else:
            pass
    # A=input()
    # print(left_data)
    # print(left_row)
    # print(left_col)
    # print(right)
    nformula = ndata
    nvary = I+J+K
    print(nformula, nvary, "{t:.3f}".format(t=time.time()-dt_now))
    lb = [0.0 for i in range(nvary)]
    ub = [np.inf for i in range(nvary)]
    for i in range(I, nvary):
        lb[i] = -np.inf
    csr_LEFT = left_lil.tocsr()
    result = lsq_linear(csr_LEFT, right, bounds=(lb, ub), verbose=2)
    X = result.x
    Cost = math.sqrt(2.0*result.cost)
    print("Finished calculating the VTEC map.",
          "{t:.3f}".format(t=time.time()-dt_now))
    # print(X)
    # A=input()

    TEC = X[0:I]
    # print(TEC[(n_TIME-1)*n_LAT*n_LON:])
    TEC_2D = TEC.reshape([n_TIME, n_LAT, n_LON])
    if smooth:
        # Space smooth
        smoothTEC = np.full((I), 0.0, dtype=float)
        for itime in range(n_TIME):
            iTEC = TEC[itime*n_LAT*n_LON:(itime+1)*n_LAT*n_LON]
            # print(iTEC)
            no_data_dict = {}
            no_data_idx = 0
            for jidx in range(n_LAT*n_LON):
                if abs(iTEC[jidx]) < 0.15:
                    no_data_dict[jidx] = no_data_idx
                    no_data_idx += 1
            L = len(no_data_dict)
            # print(L)
            H = np.full((L, L), 0.0, dtype=float)
            Hright = np.full((L, 1), 0.0, dtype=float)

            for j in range(n_LAT*n_LON):
                if not j in no_data_dict:
                    pass
                else:
                    jcount = 0
                    j_idx = no_data_dict[j]
                    j_LON = j % n_LON
                    j_LAT = int((j-j_LON)/n_LON)
                    if j_LON != 0:  # has left block
                        l_j = j-1
                        jcount += 1
                        if l_j in no_data_dict:
                            l_j_idx = no_data_dict[l_j]
                            H[j_idx, l_j_idx] = 1.0
                        else:
                            Hright[j_idx] -= iTEC[l_j]
                    if j_LON != n_LON-1:  # has right block
                        jcount += 1
                        r_j = j+1
                        if r_j in no_data_dict:
                            r_j_idx = no_data_dict[r_j]
                            H[j_idx, r_j_idx] = 1.0
                        else:
                            Hright[j_idx] -= iTEC[r_j]
                    if j_LAT != 0:  # has lower block
                        jcount += 1
                        l_j = j-n_LON
                        if l_j in no_data_dict:
                            l_j_idx = no_data_dict[l_j]
                            H[j_idx, l_j_idx] = 1.0
                        else:
                            Hright[j_idx] -= iTEC[l_j]
                    if j_LAT != n_LAT-1:  # has upper block
                        jcount += 1
                        u_j = j+n_LON
                        if u_j in no_data_dict:
                            u_j_idx = no_data_dict[u_j]
                            H[j_idx, u_j_idx] = 1.0
                        else:
                            Hright[j_idx] -= iTEC[u_j]
                    H[j_idx, j_idx] = -jcount*1.0
            # print(H)
            # print(Hright)
            ismooth = np.linalg.solve(H, Hright)
            # print(ismooth)
            # A=input()
            for k in range(n_LAT*n_LON):
                if k in no_data_dict:
                    # print(no_data_dict[k])
                    smoothTEC[itime*n_LAT*n_LON +
                              k] = ismooth[no_data_dict[k]]
                else:
                    smoothTEC[itime*n_LAT*n_LON +
                              k] = TEC[itime*n_LAT*n_LON+k]
        TEC_2D = smoothTEC.reshape([n_TIME, n_LAT, n_LON])
    SAT_bias = X[I:I+J]
    REC_bias = X[I+J:I+J+K]
    # print(len(REC_bias))
    print("Finish exporting VTEC map and bias",
          "{t:.3f}".format(t=time.time()-dt_now))
    # print(TEC)
    # print(SAT_bias)
    # print(REC_bias)

    TEC_folder = "{vtec}/{c}/{y:04}/{d:03}".format(
        vtec=vtec_folder, c=country, y=year4, d=day)
    os.makedirs(TEC_folder, exist_ok=True)
    path_nonBias_TEC = TEC_folder+"/{ver}.vtecmap".format(ver=version)
    with open(path_nonBias_TEC, "w") as pnt:
        print("# RINEX ver G_3.02", file=pnt)
        print("#", file=pnt)
        print("# RUN BY", file=pnt)
        print("# PROGRAM {p}".format(p=version), file=pnt)
        print("# UTCTIME {t}".format(t=datetime.datetime.now()), file=pnt)
        print("#", file=pnt)
        print("# This file contain VTEC, satelite bias, receiver bias", file=pnt)
        print("#", file=pnt)
        print("# LATITUDE : {m_b:.1f} ~ {M_b:.1f}".format(
            m_b=m_LAT, M_b=M_LAT), file=pnt)
        print("# LONGITUDE : {m_l:.1f} ~ {M_l:.1f}".format(
            m_l=m_LON, M_l=M_LON), file=pnt)
        print("#", file=pnt)
        print("# HEIGHT : {h:.1f} [km]".format(h=H_ipp), file=pnt)
        print("#", file=pnt)
        print("# DAY : {y:04}/{d:03}"
              .format(y=year4, d=day), file=pnt)
        print("#", file=pnt)
        print("# Region where TEC value is assumed to be constant", file=pnt)
        print("# delta LAT : {d_b:.1f}".format(
            d_b=TEC_ident_area), file=pnt)
        print("# delta LON : {d_l:.1f}".format(
            d_l=TEC_ident_area), file=pnt)
        print("# delta Epoch 1 : {d_t:04d}".format(
            d_t=TEC_ident_epoch), file=pnt)
        print("#", file=pnt)
        print(
            "# Time where Receiver and Satelite bias value is assumed to be constant", file=pnt)
        print("# delta Epoch 2 : {d_t:04d}".format(
            d_t=bias_ident_time), file=pnt)
        print("#", file=pnt)
        print("# Number of LAT : {n_b:02}".format(n_b=n_LAT), file=pnt)
        print("# Number of LON : {n_l:02}".format(n_l=n_LON), file=pnt)
        print("# Number of Satelite : {n_sat:02}".format(
            n_sat=J), file=pnt)
        print("# Number of Receiver : {n_rec:04}".format(
            n_rec=nrec), file=pnt)
        print("#", file=pnt)
        print("# Residual Value of Cost Function : {cos:+015.8f}".format(
            cos=Cost), file=pnt)
        print("# Number of Formulat : {nf:07d}".format(nf=nformula), file=pnt)
        print("#", file=pnt)
        print("# 1.VTEC [TECU]", file=pnt)
        print("# 2.Satelite Bias [TECU]", file=pnt)
        print("# 3.Receiver Bias [TECU]", file=pnt)
        print("#", file=pnt)
        print("# END OF HEADER", file=pnt)
        print("", file=pnt)
        print("# 1.VTEC [TECU]", file=pnt)
        for h in range(n_TIME):
            utc = TEC_ident_epoch*h*HOUR_PAR_EPOCH
            hour = math.floor(utc)
            subhour = int((utc-hour)*3600.0)
            minute = math.floor(subhour/60)
            second = subhour % 60
            pnt.write("Epoch : {epoch:04d} -> UTC : {ho:02}:{m:02}:{s:02}\n".format(
                epoch=TEC_ident_epoch*h, ho=hour, m=minute, s=second))
            for i in range(n_LAT):
                for j in range(n_LON):
                    pnt.write("{vtec:07.3f} ".format(vtec=TEC_2D[h, i, j]))
                pnt.write("\n")
            pnt.write("\n")
        print("", file=pnt)
        print("# 2.Satelite Bias [TECU]", file=pnt)
        for i in range(J):
            pnt.write("{sat_id:02} {sat_bias:08.3f}\n".format(
                sat_id=satelite_list[i], sat_bias=SAT_bias[i]))
        print("", file=pnt)
        print("# 3.Receiver Bias [TECU]", file=pnt)
        print("GPS/QZSS/Galileo", file=pnt)
        for i in range(nrec):
            print("{rec_id} {rec_b1:08.3f} {rec_b2:08.3f} {rec_b3:08.3f}".format(
                rec_id=station_dict_swap[i], rec_b1=REC_bias[i],
                rec_b2=REC_bias[i+nrec], rec_b3=REC_bias[i+nrec]
            ), file=pnt)
    print("{t:.2f}".format(t=time.time()-start))


def mdf2bias_v2_10(year4: int, day: int, smooth: bool):
    start = time.time()
    # station_dict[受信局ID]=割り振られた番号
    station_dict = {}
    satelite_list = []
    n_rec = 0
    for isat in range(Gmaxsat):
        modified_TEC = "{md}/{c}/{y:04}/{d:03d}/G{sat:02}".format(
            md=mdf_folder, y=year4, d=day, sat=isat, c=country)
        if os.path.isdir(modified_TEC):
            # print(modified_TEC)
            satelite_list.append("G{sat:02}".format(sat=isat))
            mod_files = glob.glob(modified_TEC+"/*.mdf")
            for mod_file in mod_files:
                # print(mod_file)
                receiver_code = mod_file[-8:-4]
                if not receiver_code in station_dict:
                    station_dict[receiver_code] = n_rec
                    n_rec += 1

    n_record = np.full((len(satelite_list)), 0, dtype=int)
    # station_dict_swap[割り振られた番号]=受信局ID
    station_dict_swap = {v: k for k, v in station_dict.items()}
    # Region where TEC value is assumed to be constant
    TEC_ident_epoch = 30  # [epoch]  0.1[hour] = 6[min] = 12[epoch]
    TEC_ident_area = 1.0  # deg
    # Time where satelite and receiver bias to be constant
    bias_ident_time = 2880  # [epoch] 2880[epoch] = 1440[min] = 24[hour]
    EPOCH_PAR_DAY = 2880
    HOUR_PAR_EPOCH = 24.0/2880

    # glonass_slot_file = "E:/mdf/{c}/{y:04d}/{d:03d}/glonass.txt".format(
    #     c=country, y=year4, d=gday)
    # with open(glonass_slot_file, "r") as f:
    #     while True:
    #         line = f.readline()
    #         if not line:
    #             break
    #         else:
    #             glonass_slot[int(line.split()[0])] = int(line.split()[1])

    m_LAT = 15.0
    M_LAT = 60.0
    m_LON = 120.0
    M_LON = 165.0

    n_LAT = round((M_LAT-m_LAT)/TEC_ident_area)
    n_LON = round((M_LON-m_LON)/TEC_ident_area)

    I = math.ceil(EPOCH_PAR_DAY/TEC_ident_epoch) * \
        n_LAT*n_LON     # Number of pixel
    J = len(satelite_list)              # Number of satelite
    nbias = 1
    # GPS ... 1
    K = len(station_dict)*nbias   # (Number of receiver) X (Number of DCB)
    nrec = len(station_dict)

    Mdata = math.ceil(EPOCH_PAR_DAY/TEC_ident_epoch)*K*20
    print("Set VTEC mapping situation. :", I, J, nrec, K, Mdata,
          "{t:.3f}".format(t=time.time()-dt_now))  # 0.121[s]

    raw_stec_data = np.full((Mdata), 0.0, dtype=float)
    raw_sat_data = np.full((Mdata, 3), 0.0, dtype=float)
    raw_rec_data = np.full((Mdata, 3), 0.0, dtype=float)
    rec_num = np.full((Mdata), 0, dtype=int)
    sat_num = np.full((Mdata), 0, dtype=int)
    time_num = np.full((Mdata), 0, dtype=int)
    sat_idxs = np.full((Mdata), 0, dtype=int)
    idx = 0
    Epochlist = []
    for i in range(round(EPOCH_PAR_DAY/TEC_ident_epoch)):
        Epochlist.append(TEC_ident_epoch*i)
    n_TIME = len(Epochlist)
    for isat in range(len(satelite_list)):
        modified_TEC = "{md}/{c}/{y:04}/{d:03d}/{sat}".format(
            md=mdf_folder, y=year4, d=day, sat=satelite_list[isat], c=country)
        satid = satelite_list[isat]
        if os.path.isdir(modified_TEC):
            mod_files = glob.glob(modified_TEC+"\*.mdf")
            for mod_file in mod_files:
                # print(mod_file)
                with open(mod_file, "r") as m_f:
                    recid = mod_file[-8:-4]
                    # Header
                    while True:
                        line = m_f.readline()
                        # print(line)
                        if "END OF HEADER" in line:
                            break
                        elif "#" in line:
                            continue
                        elif not line:
                            break
                    # Data
                    Timeidx = 0
                    while True:
                        line = m_f.readline()
                        if not line:
                            break
                        #  print(line)
                        dline = line.split()
                        # print(dline)
                        epoch = round(float(dline[0])*120.0)
                        if epoch > Epochlist[Timeidx]:
                            while epoch > Epochlist[Timeidx] and Timeidx < n_TIME-1:
                                Timeidx += 1
                        if epoch == Epochlist[Timeidx]:
                            n_record[isat] += 1
                            rawstec = float(dline[1])
                            satx = float(dline[4])
                            saty = float(dline[5])
                            satz = float(dline[6])
                            recx = float(dline[7])
                            recy = float(dline[8])
                            recz = float(dline[9])
                            raw_stec_data[idx] = rawstec
                            raw_sat_data[idx, 0] = satx
                            raw_sat_data[idx, 1] = saty
                            raw_sat_data[idx, 2] = satz
                            raw_rec_data[idx, 0] = recx
                            raw_rec_data[idx, 1] = recy
                            raw_rec_data[idx, 2] = recz
                            rec_num[idx] = station_dict[recid]
                            sat_num[idx] = isat
                            time_num[idx] = Timeidx
                            if "G" in satid:
                                sat_idxs[idx] = 0
                            Timeidx += 1
                            idx += 1
                        if Timeidx >= n_TIME:
                            break

    # for i in range(idx):
    #     print(i,":",raw_stec_data[i],raw_sat_data[i],raw_rec_data[i],rec_num[i],sat_num[i],time_num[i])
    # A=input()
    print("Number of Record", n_record)
    ndata = idx-1
    print("End importing raw stec data. Data number :", ndata,
          "{t:.3f}".format(t=time.time()-dt_now))  # 52.859[s]
    left_lil = lil_matrix((ndata, I+J+K))
    right = [0.0 for i in range(ndata)]

    for i in range(ndata):
        # G ... 0
        iband = sat_idxs[i]
        isat = sat_num[i]
        irec = rec_num[i]
        itime = time_num[i]
        iSTEC = raw_stec_data[i]
        iREC = XYZ()
        iREC.set(raw_rec_data[i, 0],
                 raw_rec_data[i, 1], raw_rec_data[i, 2])
        iSAT = XYZ()
        iSAT.set(raw_sat_data[i, 0],
                 raw_sat_data[i, 1], raw_sat_data[i, 2])
        iIPP = specify_H(iREC, iSAT, H_ipp*1.0e+3)
        i1IPP = specify_H(iREC, iSAT, H1_ipp*1.0e+3)
        i2IPP = specify_H(iREC, iSAT, H2_ipp*1.0e+3)
        S1 = (i1IPP-i2IPP).L2()
        S0 = (H2_ipp-H1_ipp)*1.0e+3
        iIPP_blh = iIPP.to_BLH()
        # print(i,":",isat,irec,itime,iSTEC,iIPP_blh)
        iIPP_b = iIPP_blh.b
        iIPP_l = iIPP_blh.l
        iIPP_b_idx = math.floor((iIPP_b-m_LAT)/TEC_ident_area)
        iIPP_l_idx = math.floor((iIPP_l-m_LON)/TEC_ident_area)
        if -1 < iIPP_b_idx < n_LAT and -1 < iIPP_l_idx < n_LON:
            iTEC_idx = iIPP_b_idx*n_LON+iIPP_l_idx
            # for VTEC
            left_lil[i, iTEC_idx+itime*n_LAT*n_LON] = S1/S0
            # for satelite bias
            left_lil[i, I+isat] = 1.0
            # for receiver bias
            left_lil[i, I+J+iband*nrec+irec] = 1.0
            # for right side
            right[i] = (iSTEC)
            # print(iIDX)
        else:
            pass
    # A=input()
    # print(left_data)
    # print(left_row)
    # print(left_col)
    # print(right)
    nformula = ndata
    nvary = I+J+K
    print(nformula, nvary, "{t:.3f}".format(t=time.time()-dt_now))
    lb = [0.0 for i in range(nvary)]
    ub = [np.inf for i in range(nvary)]
    for i in range(I, nvary):
        lb[i] = -np.inf
    csr_LEFT = left_lil.tocsr()
    result = lsq_linear(csr_LEFT, right, bounds=(lb, ub), verbose=2)
    X = result.x
    Cost = math.sqrt(2.0*result.cost)
    print("Finished calculating the VTEC map.",
          "{t:.3f}".format(t=time.time()-dt_now))
    # print(X)
    # A=input()

    TEC = X[0:I]
    # print(TEC[(n_TIME-1)*n_LAT*n_LON:])
    TEC_2D = TEC.reshape([n_TIME, n_LAT, n_LON])
    if smooth:
        # Space smooth
        smoothTEC = np.full((I), 0.0, dtype=float)
        for itime in range(n_TIME):
            iTEC = TEC[itime*n_LAT*n_LON:(itime+1)*n_LAT*n_LON]
            # print(iTEC)
            no_data_dict = {}
            no_data_idx = 0
            for jidx in range(n_LAT*n_LON):
                if abs(iTEC[jidx]) < 0.15:
                    no_data_dict[jidx] = no_data_idx
                    no_data_idx += 1
            L = len(no_data_dict)
            # print(L)
            H = np.full((L, L), 0.0, dtype=float)
            Hright = np.full((L, 1), 0.0, dtype=float)

            for j in range(n_LAT*n_LON):
                if not j in no_data_dict:
                    pass
                else:
                    jcount = 0
                    j_idx = no_data_dict[j]
                    j_LON = j % n_LON
                    j_LAT = int((j-j_LON)/n_LON)
                    if j_LON != 0:  # has left block
                        l_j = j-1
                        jcount += 1
                        if l_j in no_data_dict:
                            l_j_idx = no_data_dict[l_j]
                            H[j_idx, l_j_idx] = 1.0
                        else:
                            Hright[j_idx] -= iTEC[l_j]
                    if j_LON != n_LON-1:  # has right block
                        jcount += 1
                        r_j = j+1
                        if r_j in no_data_dict:
                            r_j_idx = no_data_dict[r_j]
                            H[j_idx, r_j_idx] = 1.0
                        else:
                            Hright[j_idx] -= iTEC[r_j]
                    if j_LAT != 0:  # has lower block
                        jcount += 1
                        l_j = j-n_LON
                        if l_j in no_data_dict:
                            l_j_idx = no_data_dict[l_j]
                            H[j_idx, l_j_idx] = 1.0
                        else:
                            Hright[j_idx] -= iTEC[l_j]
                    if j_LAT != n_LAT-1:  # has upper block
                        jcount += 1
                        u_j = j+n_LON
                        if u_j in no_data_dict:
                            u_j_idx = no_data_dict[u_j]
                            H[j_idx, u_j_idx] = 1.0
                        else:
                            Hright[j_idx] -= iTEC[u_j]
                    H[j_idx, j_idx] = -jcount*1.0
            # print(H)
            # print(Hright)
            ismooth = np.linalg.solve(H, Hright)
            # print(ismooth)
            # A=input()
            for k in range(n_LAT*n_LON):
                if k in no_data_dict:
                    # print(no_data_dict[k])
                    smoothTEC[itime*n_LAT*n_LON +
                              k] = ismooth[no_data_dict[k]]
                else:
                    smoothTEC[itime*n_LAT*n_LON +
                              k] = TEC[itime*n_LAT*n_LON+k]
        TEC_2D = smoothTEC.reshape([n_TIME, n_LAT, n_LON])
    SAT_bias = X[I:I+J]
    REC_bias = X[I+J:I+J+K]
    # print(len(REC_bias))
    print("Finish exporting VTEC map and bias",
          "{t:.3f}".format(t=time.time()-dt_now))
    # print(TEC)
    # print(SAT_bias)
    # print(REC_bias)

    TEC_folder = "{vtec}/{c}/{y:04}/{d:03}".format(
        vtec=vtec_folder, c=country, y=year4, d=day)
    os.makedirs(TEC_folder, exist_ok=True)
    path_nonBias_TEC = TEC_folder+"/{ver}.vtecmap".format(ver=version)
    with open(path_nonBias_TEC, "w") as pnt:
        print("# RINEX ver G_3.02", file=pnt)
        print("#", file=pnt)
        print("# RUN BY", file=pnt)
        print("# PROGRAM {p}".format(p=version), file=pnt)
        print("# UTCTIME {t}".format(t=datetime.datetime.now()), file=pnt)
        print("#", file=pnt)
        print("# This file contain VTEC, satelite bias, receiver bias", file=pnt)
        print("#", file=pnt)
        print("# LATITUDE : {m_b:.1f} ~ {M_b:.1f}".format(
            m_b=m_LAT, M_b=M_LAT), file=pnt)
        print("# LONGITUDE : {m_l:.1f} ~ {M_l:.1f}".format(
            m_l=m_LON, M_l=M_LON), file=pnt)
        print("#", file=pnt)
        print("# HEIGHT : {h:.1f} [km]".format(h=H_ipp), file=pnt)
        print("#", file=pnt)
        print("# DAY : {y:04}/{d:03}"
              .format(y=year4, d=day), file=pnt)
        print("#", file=pnt)
        print("# Region where TEC value is assumed to be constant", file=pnt)
        print("# delta LAT : {d_b:.1f}".format(
            d_b=TEC_ident_area), file=pnt)
        print("# delta LON : {d_l:.1f}".format(
            d_l=TEC_ident_area), file=pnt)
        print("# delta Epoch 1 : {d_t:04d}".format(
            d_t=TEC_ident_epoch), file=pnt)
        print("#", file=pnt)
        print(
            "# Time where Receiver and Satelite bias value is assumed to be constant", file=pnt)
        print("# delta Epoch 2 : {d_t:04d}".format(
            d_t=bias_ident_time), file=pnt)
        print("#", file=pnt)
        print("# Number of LAT : {n_b:02}".format(n_b=n_LAT), file=pnt)
        print("# Number of LON : {n_l:02}".format(n_l=n_LON), file=pnt)
        print("# Number of Satelite : {n_sat:02}".format(
            n_sat=J), file=pnt)
        print("# Number of Receiver : {n_rec:04}".format(
            n_rec=nrec), file=pnt)
        print("#", file=pnt)
        print("# Residual Value of Cost Function : {cos:+015.8f}".format(
            cos=Cost), file=pnt)
        print("# Number of Formulat : {nf:07d}".format(nf=nformula), file=pnt)
        print("#", file=pnt)
        print("# 1.VTEC [TECU]", file=pnt)
        print("# 2.Satelite Bias [TECU]", file=pnt)
        print("# 3.Receiver Bias [TECU]", file=pnt)
        print("#", file=pnt)
        print("# END OF HEADER", file=pnt)
        print("", file=pnt)
        print("# 1.VTEC [TECU]", file=pnt)
        for h in range(n_TIME):
            utc = TEC_ident_epoch*h*HOUR_PAR_EPOCH
            hour = math.floor(utc)
            subhour = int((utc-hour)*3600.0)
            minute = math.floor(subhour/60)
            second = subhour % 60
            pnt.write("Epoch : {epoch:04d} -> UTC : {ho:02}:{m:02}:{s:02}\n".format(
                epoch=TEC_ident_epoch*h, ho=hour, m=minute, s=second))
            for i in range(n_LAT):
                for j in range(n_LON):
                    pnt.write("{vtec:07.3f} ".format(vtec=TEC_2D[h, i, j]))
                pnt.write("\n")
            pnt.write("\n")
        print("", file=pnt)
        print("# 2.Satelite Bias [TECU]", file=pnt)
        for i in range(J):
            pnt.write("{sat_id:02} {sat_bias:08.3f}\n".format(
                sat_id=satelite_list[i], sat_bias=SAT_bias[i]))
        print("", file=pnt)
        print("# 3.Receiver Bias [TECU]", file=pnt)
        print("GPS/QZSS/Galileo", file=pnt)
        for i in range(nrec):
            print("{rec_id} {rec_b1:08.3f} {rec_b2:08.3f} {rec_b3:08.3f}".format(
                rec_id=station_dict_swap[i], rec_b1=REC_bias[i],
                rec_b2=0.0, rec_b3=0.0
            ), file=pnt)
    print("{t:.2f}".format(t=time.time()-start))


def mdf2bias_v2_12(year4: int, day: int, smooth: bool):
    start = time.time()
    # station_dict[受信局ID]=割り振られた番号
    station_dict = {}
    satelite_list = []
    n_rec = 0
    for isat in range(Gmaxsat):
        modified_TEC = "{md}/{c}/{y:04}/{d:03d}/G{sat:02}".format(
            md=mdf_folder, y=year4, d=day, sat=isat, c=country)
        if os.path.isdir(modified_TEC):
            # print(modified_TEC)
            satelite_list.append("G{sat:02}".format(sat=isat))
            mod_files = glob.glob(modified_TEC+"/*.mdf")
            for mod_file in mod_files:
                # print(mod_file)
                receiver_code = mod_file[-8:-4]
                if not receiver_code in station_dict:
                    station_dict[receiver_code] = n_rec
                    n_rec += 1
    for isat in range(Emaxsat):
        modified_TEC = "{md}/{c}/{y:04}/{d:03d}/E{sat:02}".format(
            md=mdf_folder, y=year4, d=day, sat=isat, c=country)
        if os.path.isdir(modified_TEC):
            # print(modified_TEC)
            satelite_list.append("E{sat:02}".format(sat=isat))
            mod_files = glob.glob(modified_TEC+"/*.mdf")
            for mod_file in mod_files:
                # print(mod_file)
                receiver_code = mod_file[-8:-4]
                if not receiver_code in station_dict:
                    station_dict[receiver_code] = n_rec
                    n_rec += 1
    for isat in range(Jmaxsat):
        modified_TEC = "{md}/{c}/{y:04}/{d:03d}/J{sat:02}".format(
            md=mdf_folder, y=year4, d=day, sat=isat, c=country)
        if os.path.isdir(modified_TEC):
            # print(modified_TEC)
            satelite_list.append("J{sat:02}".format(sat=isat))
            mod_files = glob.glob(modified_TEC+"/*.mdf")
            for mod_file in mod_files:
                # print(mod_file)
                receiver_code = mod_file[-8:-4]
                if not receiver_code in station_dict:
                    station_dict[receiver_code] = n_rec
                    n_rec += 1
    # for isat in range(Rmaxsat):
    #     modified_TEC = "E:/modified_data/{c}/{y:04}/{d:03d}/R{sat:02}".format(
    #         y=year4, d=day, sat=isat, c=country)
    #     if os.path.isdir(modified_TEC):
    #         # print(modified_TEC)
    #         satelite_list.append("R{sat:02}".format(sat=isat))
    #         mod_files = glob.glob(modified_TEC+"/*.mdf")
    #         for mod_file in mod_files:
    #             # print(mod_file)
    #             receiver_code = mod_file[-8:-4]
    #             if not receiver_code in station_dict:
    #                 station_dict[receiver_code] = n_rec
    #                 n_rec += 1
    n_record = np.full((len(satelite_list)), 0, dtype=int)
    # station_dict_swap[割り振られた番号]=受信局ID
    station_dict_swap = {v: k for k, v in station_dict.items()}
    # Region where TEC value is assumed to be constant
    TEC_ident_epoch = 30  # [epoch]  0.1[hour] = 6[min] = 12[epoch]
    TEC_ident_area = 1.0  # deg
    # Time where satelite and receiver bias to be constant
    bias_ident_time = 2880  # [epoch] 2880[epoch] = 1440[min] = 24[hour]
    EPOCH_PAR_DAY = 2880
    HOUR_PAR_EPOCH = 24.0/2880

    m_LAT = 15.0
    M_LAT = 60.0
    m_LON = 120.0
    M_LON = 165.0

    n_LAT = round((M_LAT-m_LAT)/TEC_ident_area)
    n_LON = round((M_LON-m_LON)/TEC_ident_area)

    I = math.ceil(EPOCH_PAR_DAY/TEC_ident_epoch) * \
        n_LAT*n_LON     # Number of pixel
    J = len(satelite_list)              # Number of satelite
    nbias = 3
    # GPS ... 1
    # Galileo ... 1
    # QZSS ... 1
    # GLONASS ... 14(-07 -> 06)
    K = len(station_dict)*nbias   # (Number of receiver) X (Number of DCB)
    nrec = len(station_dict)

    Mdata = math.ceil(EPOCH_PAR_DAY/TEC_ident_epoch)*K*20
    print("Set VTEC mapping situation. :", I, J, nrec, K, Mdata,
          "{t:.3f}".format(t=time.time()-dt_now))  # 0.121[s]

    raw_stec_data = np.full((Mdata), 0.0, dtype=float)
    raw_sat_data = np.full((Mdata, 3), 0.0, dtype=float)
    raw_rec_data = np.full((Mdata, 3), 0.0, dtype=float)
    rec_num = np.full((Mdata), 0, dtype=int)
    sat_num = np.full((Mdata), 0, dtype=int)
    time_num = np.full((Mdata), 0, dtype=int)
    sat_idxs = np.full((Mdata), 0, dtype=int)
    idx = 0
    Epochlist = []
    for i in range(round(EPOCH_PAR_DAY/TEC_ident_epoch)):
        Epochlist.append(TEC_ident_epoch*i)
    n_TIME = len(Epochlist)
    for isat in range(len(satelite_list)):
        modified_TEC = "{md}/{c}/{y:04}/{d:03d}/{sat}".format(
            md=mdf_folder, y=year4, d=day, sat=satelite_list[isat], c=country)
        satid = satelite_list[isat]
        if os.path.isdir(modified_TEC):
            mod_files = glob.glob(modified_TEC+"\*.mdf")
            for mod_file in mod_files:
                # print(mod_file)
                with open(mod_file, "r") as m_f:
                    recid = mod_file[-8:-4]
                    # Header
                    while True:
                        line = m_f.readline()
                        # print(line)
                        if "END OF HEADER" in line:
                            break
                        elif "#" in line:
                            continue
                        elif not line:
                            break
                    # Data
                    Timeidx = 0
                    while True:
                        line = m_f.readline()
                        if not line:
                            break
                        #  print(line)
                        dline = line.split()
                        # print(dline)
                        epoch = round(float(dline[0])*120.0)
                        if epoch > Epochlist[Timeidx]:
                            while epoch > Epochlist[Timeidx] and Timeidx < n_TIME-1:
                                Timeidx += 1
                        if epoch == Epochlist[Timeidx]:
                            n_record[isat] += 1
                            rawstec = float(dline[1])
                            satx = float(dline[4])
                            saty = float(dline[5])
                            satz = float(dline[6])
                            recx = float(dline[7])
                            recy = float(dline[8])
                            recz = float(dline[9])
                            raw_stec_data[idx] = rawstec
                            raw_sat_data[idx, 0] = satx
                            raw_sat_data[idx, 1] = saty
                            raw_sat_data[idx, 2] = satz
                            raw_rec_data[idx, 0] = recx
                            raw_rec_data[idx, 1] = recy
                            raw_rec_data[idx, 2] = recz
                            rec_num[idx] = station_dict[recid]
                            sat_num[idx] = isat
                            time_num[idx] = Timeidx
                            if "G" in satid:
                                sat_idxs[idx] = 0
                            elif "J" in satid:
                                sat_idxs[idx] = 1
                            elif "E" in satid:
                                sat_idxs[idx] = 2
                            Timeidx += 1
                            idx += 1
                        if Timeidx >= n_TIME:
                            break

    # for i in range(idx):
    #     print(i,":",raw_stec_data[i],raw_sat_data[i],raw_rec_data[i],rec_num[i],sat_num[i],time_num[i])
    # A=input()
    print("Number of Record", n_record)
    ndata = idx-1
    print("End importing raw stec data. Data number :", ndata,
          "{t:.3f}".format(t=time.time()-dt_now))  # 52.859[s]
    left_lil = lil_matrix((ndata, I+J+K))
    right = [0.0 for i in range(ndata)]

    for i in range(ndata):
        # G ... 0 , J ... 1,E ... 2
        iband = sat_idxs[i]
        isat = sat_num[i]
        irec = rec_num[i]
        itime = time_num[i]
        iSTEC = raw_stec_data[i]
        iREC = XYZ()
        iREC.set(raw_rec_data[i, 0],
                 raw_rec_data[i, 1], raw_rec_data[i, 2])
        iSAT = XYZ()
        iSAT.set(raw_sat_data[i, 0],
                 raw_sat_data[i, 1], raw_sat_data[i, 2])
        iIPP = specify_H(iREC, iSAT, H_ipp*1.0e+3)
        i1IPP = specify_H(iREC, iSAT, H1_ipp*1.0e+3)
        i2IPP = specify_H(iREC, iSAT, H2_ipp*1.0e+3)
        S1 = (i1IPP-i2IPP).L2()
        S0 = (H2_ipp-H1_ipp)*1.0e+3
        iIPP_blh = iIPP.to_BLH()
        # print(i,":",isat,irec,itime,iSTEC,iIPP_blh)
        iIPP_b = iIPP_blh.b
        iIPP_l = iIPP_blh.l
        iIPP_b_idx = math.floor((iIPP_b-m_LAT)/TEC_ident_area)
        iIPP_l_idx = math.floor((iIPP_l-m_LON)/TEC_ident_area)
        if -1 < iIPP_b_idx < n_LAT and -1 < iIPP_l_idx < n_LON:
            iTEC_idx = iIPP_b_idx*n_LON+iIPP_l_idx
            # for VTEC
            left_lil[i, iTEC_idx+itime*n_LAT*n_LON] = S1/S0
            # for satelite bias
            left_lil[i, I+isat] = 1.0
            # for receiver bias
            left_lil[i, I+J+iband*nrec+irec] = 1.0
            # for right side
            right[i] = (iSTEC)
            # print(iIDX)
        else:
            pass
    # A=input()
    # print(left_data)
    # print(left_row)
    # print(left_col)
    # print(right)
    nformula = ndata
    nvary = I+J+K
    print(nformula, nvary, "{t:.3f}".format(t=time.time()-dt_now))
    lb = [0.0 for i in range(nvary)]
    ub = [np.inf for i in range(nvary)]
    for i in range(I, nvary):
        lb[i] = -np.inf
    csr_LEFT = left_lil.tocsr()
    result = lsq_linear(csr_LEFT, right, bounds=(lb, ub), verbose=2)
    X = result.x
    Cost = math.sqrt(2.0*result.cost)
    print("Finished calculating the VTEC map.",
          "{t:.3f}".format(t=time.time()-dt_now))
    # print(X)
    # A=input()

    TEC = X[0:I]
    # print(TEC[(n_TIME-1)*n_LAT*n_LON:])
    TEC_2D = TEC.reshape([n_TIME, n_LAT, n_LON])
    if smooth:
        # Space smooth
        smoothTEC = np.full((I), 0.0, dtype=float)
        for itime in range(n_TIME):
            iTEC = TEC[itime*n_LAT*n_LON:(itime+1)*n_LAT*n_LON]
            # print(iTEC)
            no_data_dict = {}
            no_data_idx = 0
            for jidx in range(n_LAT*n_LON):
                if abs(iTEC[jidx]) < 0.15:
                    no_data_dict[jidx] = no_data_idx
                    no_data_idx += 1
            L = len(no_data_dict)
            # print(L)
            H = np.full((L, L), 0.0, dtype=float)
            Hright = np.full((L, 1), 0.0, dtype=float)

            for j in range(n_LAT*n_LON):
                if not j in no_data_dict:
                    pass
                else:
                    jcount = 0
                    j_idx = no_data_dict[j]
                    j_LON = j % n_LON
                    j_LAT = int((j-j_LON)/n_LON)
                    if j_LON != 0:  # has left block
                        l_j = j-1
                        jcount += 1
                        if l_j in no_data_dict:
                            l_j_idx = no_data_dict[l_j]
                            H[j_idx, l_j_idx] = 1.0
                        else:
                            Hright[j_idx] -= iTEC[l_j]
                    if j_LON != n_LON-1:  # has right block
                        jcount += 1
                        r_j = j+1
                        if r_j in no_data_dict:
                            r_j_idx = no_data_dict[r_j]
                            H[j_idx, r_j_idx] = 1.0
                        else:
                            Hright[j_idx] -= iTEC[r_j]
                    if j_LAT != 0:  # has lower block
                        jcount += 1
                        l_j = j-n_LON
                        if l_j in no_data_dict:
                            l_j_idx = no_data_dict[l_j]
                            H[j_idx, l_j_idx] = 1.0
                        else:
                            Hright[j_idx] -= iTEC[l_j]
                    if j_LAT != n_LAT-1:  # has upper block
                        jcount += 1
                        u_j = j+n_LON
                        if u_j in no_data_dict:
                            u_j_idx = no_data_dict[u_j]
                            H[j_idx, u_j_idx] = 1.0
                        else:
                            Hright[j_idx] -= iTEC[u_j]
                    H[j_idx, j_idx] = -jcount*1.0
            # print(H)
            # print(Hright)
            ismooth = np.linalg.solve(H, Hright)
            # print(ismooth)
            # A=input()
            for k in range(n_LAT*n_LON):
                if k in no_data_dict:
                    # print(no_data_dict[k])
                    smoothTEC[itime*n_LAT*n_LON +
                              k] = ismooth[no_data_dict[k]]
                else:
                    smoothTEC[itime*n_LAT*n_LON +
                              k] = TEC[itime*n_LAT*n_LON+k]
        TEC_2D = smoothTEC.reshape([n_TIME, n_LAT, n_LON])
    SAT_bias = X[I:I+J]
    REC_bias = X[I+J:I+J+K]
    # print(len(REC_bias))
    print("Finish exporting VTEC map and bias",
          "{t:.3f}".format(t=time.time()-dt_now))
    # print(TEC)
    # print(SAT_bias)
    # print(REC_bias)

    TEC_folder = "{vtec}/{c}/{y:04}/{d:03}".format(
        vtec=vtec_folder, c=country, y=year4, d=day)
    os.makedirs(TEC_folder, exist_ok=True)
    path_nonBias_TEC = TEC_folder+"/{ver}.vtecmap".format(ver=version)
    with open(path_nonBias_TEC, "w") as pnt:
        print("# RINEX ver G_3.02", file=pnt)
        print("#", file=pnt)
        print("# RUN BY", file=pnt)
        print("# PROGRAM {p}".format(p=version), file=pnt)
        print("# UTCTIME {t}".format(t=datetime.datetime.now()), file=pnt)
        print("#", file=pnt)
        print("# This file contain VTEC, satelite bias, receiver bias", file=pnt)
        print("#", file=pnt)
        print("# LATITUDE : {m_b:.1f} ~ {M_b:.1f}".format(
            m_b=m_LAT, M_b=M_LAT), file=pnt)
        print("# LONGITUDE : {m_l:.1f} ~ {M_l:.1f}".format(
            m_l=m_LON, M_l=M_LON), file=pnt)
        print("#", file=pnt)
        print("# HEIGHT : {h:.1f} [km]".format(h=H_ipp), file=pnt)
        print("#", file=pnt)
        print("# DAY : {y:04}/{d:03}"
              .format(y=year4, d=day), file=pnt)
        print("#", file=pnt)
        print("# Region where TEC value is assumed to be constant", file=pnt)
        print("# delta LAT : {d_b:.1f}".format(
            d_b=TEC_ident_area), file=pnt)
        print("# delta LON : {d_l:.1f}".format(
            d_l=TEC_ident_area), file=pnt)
        print("# delta Epoch 1 : {d_t:04d}".format(
            d_t=TEC_ident_epoch), file=pnt)
        print("#", file=pnt)
        print(
            "# Time where Receiver and Satelite bias value is assumed to be constant", file=pnt)
        print("# delta Epoch 2 : {d_t:04d}".format(
            d_t=bias_ident_time), file=pnt)
        print("#", file=pnt)
        print("# Number of LAT : {n_b:02}".format(n_b=n_LAT), file=pnt)
        print("# Number of LON : {n_l:02}".format(n_l=n_LON), file=pnt)
        print("# Number of Satelite : {n_sat:02}".format(
            n_sat=J), file=pnt)
        print("# Number of Receiver : {n_rec:04}".format(
            n_rec=nrec), file=pnt)
        print("#", file=pnt)
        print("# Residual Value of Cost Function : {cos:+015.8f}".format(
            cos=Cost), file=pnt)
        print("# Number of Formulat : {nf:07d}".format(nf=nformula), file=pnt)
        print("#", file=pnt)
        print("# 1.VTEC [TECU]", file=pnt)
        print("# 2.Satelite Bias [TECU]", file=pnt)
        print("# 3.Receiver Bias [TECU]", file=pnt)
        print("#", file=pnt)
        print("# END OF HEADER", file=pnt)
        print("", file=pnt)
        print("# 1.VTEC [TECU]", file=pnt)
        for h in range(n_TIME):
            utc = TEC_ident_epoch*h*HOUR_PAR_EPOCH
            hour = math.floor(utc)
            subhour = int((utc-hour)*3600.0)
            minute = math.floor(subhour/60)
            second = subhour % 60
            pnt.write("Epoch : {epoch:04d} -> UTC : {ho:02}:{m:02}:{s:02}\n".format(
                epoch=TEC_ident_epoch*h, ho=hour, m=minute, s=second))
            for i in range(n_LAT):
                for j in range(n_LON):
                    pnt.write("{vtec:07.3f} ".format(vtec=TEC_2D[h, i, j]))
                pnt.write("\n")
            pnt.write("\n")
        print("", file=pnt)
        print("# 2.Satelite Bias [TECU]", file=pnt)
        for i in range(J):
            pnt.write("{sat_id:02} {sat_bias:08.3f}\n".format(
                sat_id=satelite_list[i], sat_bias=SAT_bias[i]))
        print("", file=pnt)
        print("# 3.Receiver Bias [TECU]", file=pnt)
        print("GPS(L1-L2)/QZSS/Galileo(L1-L5)", file=pnt)
        for i in range(nrec):
            pnt.write("{rec_id} ".format(rec_id=station_dict_swap[i]))
            for j in range(nbias):
                pnt.write("{rec_bias:08.3f} ".format(
                    rec_bias=REC_bias[i+j*nrec]))
            pnt.write("\n")
    print("{t:.2f}".format(t=time.time()-start))


def mdf2bias(year4: int, day: int, smooth: bool, ver: str):
    if ver == "v2_12":
        return mdf2bias_v2_12(year4, day, smooth)
    elif ver == "v2_10":
        return mdf2bias_v2_10(year4, day, smooth)
    elif ver == "v3_02":
        return mdf2bias_v3_02(year4, day, smooth)


def mdf2vtc(year4: int, day: int):
    bias_file = "{vtec}/{c}/{y:04}/{d:03}/{v}.vtecmap".format(
        vtec=vtec_folder, c=country, y=year4, d=day, v=version)

    sat_biases = {}
    rec_biases = {}

    with open(bias_file, "r") as bf:
        nsat = 0
        nrec = 0
        while True:
            ln = bf.readline()
            if "Number of Satelite" in ln:
                nsat = int(ln.split()[5])
            elif "Number of Receiver" in ln:
                nrec = int(ln.split()[5])
            if "END OF HEADER" in ln:
                break

        while True:
            ln = bf.readline()
            if "Satelite Bias" in ln:
                break

        for isat in range(nsat):
            ln = bf.readline()
            satid = ln.split()[0]
            satbias = float(ln.split()[1])
            sat_biases[satid] = satbias

        while True:
            ln = bf.readline()
            if "Receiver Bias" in ln:
                break
        ln = bf.readline()

        for irec in range(nrec):
            ln = bf.readline()
            recid = ln.split()[0]
            rec_biases[recid] = []
            for i in range(3):
                rec_biases[recid].append(float(ln.split()[i+1]))

    # glonass_slot_file = "E:/mdf/{c}/{y:04d}/{d:03d}/glonass.txt".format(
    #     c=country, y=year4, d=gday)
    # with open(glonass_slot_file, "r") as f:
    #     while True:
    #         line = f.readline()
    #         if not line:
    #             break
    #         else:
    #             glonass_slot[int(line.split()[0])] = int(line.split()[1])
    #  print(rec_biases)
    mod_folder = "{md}/{c}/{y:04}/{d:03}".format(
        md=mdf_folder, c=country, y=year4, d=day)
    mod_sat_folders = glob.glob(mod_folder+"/*")

    for mod_sat_folder in mod_sat_folders:
        if not os.path.isdir(mod_sat_folder):
            continue
        sat_id = mod_sat_folder[-3:]
        sat_bias = sat_biases[sat_id]
        mod_files = glob.glob(mod_sat_folder+"/*.mdf")
        for mod_file in mod_files:
            rec_id = mod_file[-8:-4]
            if "G" in sat_id:
                rec_bias = rec_biases[rec_id][0]
            elif "J" in sat_id:
                rec_bias = rec_biases[rec_id][1]
            elif "E" in sat_id:
                rec_bias = rec_biases[rec_id][2]
            # elif "R" in sat_id:
            #     sat_code = int(sat_id[1:3])
            #     rec_bias = rec_biases[rec_id][glonass_slot[sat_code]+9]
            # print(sat_id, rec_id, ":", sat_bias, rec_bias)
            with open(mod_file, "r") as mf:
                lines = [s.strip() for s in mf.readlines()]
            slines = [line.split() for line in lines]

            if len(lines) < 1:
                continue

            kheader = 0
            for iln in range(len(lines)):
                if "END OF HEADER" in lines[iln]:
                    kheader = iln
                    break

            no_bias_file = "{ubs}/{c}/{y:04}/{d:03}/{sat}/{rec}.teq".format(
                ubs=unbias_folder, c=country, y=year4, d=day, sat=sat_id, rec=rec_id)
            os.makedirs("{ubs}/{c}/{y:04}/{d:03}/{sat}/".format(
                ubs=unbias_folder, c=country, y=year4, d=day, sat=sat_id), exist_ok=True)
            with open(no_bias_file, "w") as nbf:
                idx = 0
                while True:
                    if "END OF HEADER" in lines[idx]:
                        break
                    idx += 1

                print("# RUN BY", file=nbf)
                print("# PROGRAM {p}".format(p=version), file=nbf)
                print("# UTCTIME {t}".format(
                    t=datetime.datetime.now()), file=nbf)
                print("#", file=nbf)
                print("# WAVE FREQUENCY", file=nbf)
                if "G" in sat_id:
                    print("# L1 {l1:d}".format(l1=f_l1_GPS), file=nbf)
                    print("# L2 {l2:d}".format(l2=f_l2_GPS), file=nbf)
                if "J" in sat_id:
                    print("# L1 {l1:d}".format(l1=f_l1_QZSS), file=nbf)
                    print("# L5 {l2:d}".format(l2=f_l5_QZSS), file=nbf)
                if "E" in sat_id:
                    print("# L1 {l1:d}".format(l1=f_l1_Galileo), file=nbf)
                    print("# L5 {l2:d}".format(l2=f_l5_Galileo), file=nbf)
                if "R" in sat_id:
                    sat_code = int(sat_id[1:3])
                    print("# L1 {l1:d}".format(l1=f_l1_GLONASS_base +
                          glonass_slot[sat_code]*f_l1_GLONASS_del), file=nbf)
                    print("# L2 {l2:d}".format(l2=f_l2_GLONASS_base +
                          glonass_slot[sat_code]*f_l2_GLONASS_del), file=nbf)
                print("#", file=nbf)
                print("# 1. TIME [UTC:hour]", file=nbf)
                print("# 2. UNBIAS TEC [TECU]", file=nbf)
                print("# 3. SATELITE POSITION [m]", file=nbf)
                print("# 4.RECEIVER POSITION [m]", file=nbf)
                print("#", file=nbf)
                print("# Satelite bias : {sb:08.3f}".format(
                    sb=sat_bias), file=nbf)
                print("# Receiver bias : {rb:08.3f}".format(
                    rb=rec_bias), file=nbf)
                print("#", file=nbf)
                print("# END OF HEADER", file=nbf)

                for krec in range(kheader+1, len(lines)):
                    ktime = float(slines[krec][0])
                    kstec = float(slines[krec][1])-sat_bias-rec_bias
                    ksatx = float(slines[krec][4])
                    ksaty = float(slines[krec][5])
                    ksatz = float(slines[krec][6])
                    krecx = float(slines[krec][7])
                    krecy = float(slines[krec][8])
                    krecz = float(slines[krec][9])
                    print("{t:07.4f} {s:08.4f} {sx:013.3f} {sy:013.3f} {sz:013.3f} {rx:013.3f} {ry:013.3f} {rz:013.3f}".format(
                        t=ktime, s=kstec, sx=ksatx, sy=ksaty, sz=ksatz, rx=krecx, ry=krecy, rz=krecz
                    ), file=nbf)
        print("{y} {d} {s} finish.".format(y=year4, d=day, s=sat_id))


dt_now = time.time()

global gday

for day in days:
    gday = day
    md = datetime.date(year4, 1, 1)+datetime.timedelta(day-1)

    month = md.month
    dayofmonth = md.day

    obs_day = datetime.datetime(year4, md.month, md.day)

    path_tec = "{tec}\\{c}\\{y:04}\\{d:03}\\".format(
        tec=tec_folder, y=year4, d=day, c=country)

    GPS_list = []
    files = glob.glob(path_tec+"/*".format(y2=year2))
    stas = set()
    for file in files:
        # print(file)

        sta = file[-12:-8]
        sta = sta.lower()
        # print(sta)
        stas.add(sta)

    stations = sorted(stas)
    # print(stations)
    # rnx2mdf(year4, day, stations, "v2_12")

    mdf2bias(year4, day, False, "v2_12")

    mdf2vtc(year4, day)
