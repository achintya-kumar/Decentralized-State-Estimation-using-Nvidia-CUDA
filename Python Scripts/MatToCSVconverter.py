
import numpy
import scipy.io as sio
size = 30
path = "C:\\Users\\kumar\\Desktop\\1500"
KKT = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\KKT.mat").get('KKTb'), order='C'))
numpy.savetxt(path + "\\KKT.csv", KKT, delimiter=",", fmt='%.17f')

YT = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\YT.mat").get('YTb'), order='C'))
numpy.savetxt(path + "\\YT.csv", YT, delimiter=",", fmt='%.17f')

YT_real = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\YT_real.mat").get('YT_real'), order='C'))
numpy.savetxt(path + "\\YT_real.csv", YT_real, delimiter=",", fmt='%.17f')

YT_imag = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\YT_imag.mat").get('YT_imag'), order='C'))
numpy.savetxt(path + "\\YT_imag.csv", YT_imag, delimiter=",", fmt='%.17f')

YKK = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\YKK.mat").get('YKKb'), order='C'))
numpy.savetxt(path + "\\YKK.csv", YKK, delimiter=",", fmt='%.17f')

YKK_real = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\YKK_real.mat").get('YKK_real'), order='C'))
numpy.savetxt(path + "\\YKK_real.csv", YKK_real, delimiter=",", fmt='%.17f')

YKK_imag = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\YKK_imag.mat").get('YKK_imag'), order='C'))
numpy.savetxt(path + "\\YKK_imag.csv", YKK_imag, delimiter=",", fmt='%.17f')

INIT = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\INIT.mat").get('uK0b'), order='C'))
numpy.savetxt(path + "\\INIT.csv", INIT, delimiter=",", fmt='%.17f')

MEAS = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\MEAS.mat").get('MEASb'), order='C'))
numpy.savetxt(path + "\\MEAS.csv", MEAS, delimiter=",", fmt='%.17f')

z_uT = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\z_uT.mat").get('z_uTb'), order='C'))
numpy.savetxt(path + "\\z_uT.csv", z_uT, delimiter=",", fmt='%.17f')

z_pT = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\z_pT.mat").get('z_pTb'), order='C'))
numpy.savetxt(path + "\\z_pT.csv", z_pT, delimiter=",", fmt='%.17f')

z_qT = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\z_qT.mat").get('z_qTb'), order='C'))
numpy.savetxt(path + "\\z_qT.csv", z_qT, delimiter=",", fmt='%.17f')

z_pK = numpy.asmatrix(numpy.array(sio.loadmat(path+ "\\z_pK.mat").get('z_pKb'), order='C'))
numpy.savetxt(path + "\\z_pK.csv", z_pK, delimiter=",", fmt='%.17f')

z_qK = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\z_qK.mat").get('z_qKb'), order='C'))
numpy.savetxt(path + "\\z_qK.csv", z_qK, delimiter=",", fmt='%.17f')

z_uTabs = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\z_uTabs.mat").get('z_uTabsb'), order='C'))
numpy.savetxt(path + "\\z_uTabs.csv", z_uTabs, delimiter=",", fmt='%.17f')

z_uTang = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\z_uTang.mat").get('z_uTangb'), order='C'))
numpy.savetxt(path + "\\z_uTang.csv", z_uTang, delimiter=",", fmt='%.17f')

z_iTabs = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\z_iTabs.mat").get('z_iTabsb'), order='C'))
numpy.savetxt(path + "\\z_iTabs.csv", z_iTabs, delimiter=",", fmt='%.17f')

z_iTang = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\z_iTang.mat").get('z_iTangb'), order='C'))
numpy.savetxt(path + "\\z_iTang.csv", z_iTang, delimiter=",", fmt='%.17f')

voltages = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\voltages.mat").get('voltages'), order='C'))
numpy.savetxt(path + "\\voltages.csv", voltages, delimiter=",", fmt='%.17f')

angles = numpy.asmatrix(numpy.array(sio.loadmat(path + "\\angles.mat").get('angles'), order='C'))
numpy.savetxt(path + "\\angles.csv", angles, delimiter=",", fmt='%.17f')
