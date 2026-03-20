import asyncio
import os
import sys

import numpy as np
import pygame
from pygame.locals import *

import Three_States as dm

ICONSIZE = 30 # size of the tokens and individual board spaces in pixels

FPS = 20 # frames per second to update the screen
WINDOWWIDTH = int(1.4*917) # width of the program's window, in pixels
WINDOWHEIGHT = int(1.4*582) # height in pixels

BRIGHTBLUE = (0, 50, 255)
RED = (255,64,64)
WHITE = (255, 255, 255)
BLACK = (0,0,0)

RIGHT = 3
LEFT = 1

TEXTCOLOR = WHITE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BGMAP = None
BGTRACT = None

MED = 'med'
FIRE = 'fire'
JOINT = 'joint'

UPLEFT = (44.988032, -93.207538)
UPLEFT_PIX = (75,43)
DOWNRIGHT = (44.890738, -93.004278)
DOWNRIGHT_PIX = (1140,768)

tracts_pix = [(52,27),(283,9),(386,5),(498,7),(607,11),(700,5),(925,7),(505,83),(702,86),(768,71),(820,60)
                 ,(390,130),(502,160),(610,117),(719,165),(820,141),(851,118),(987,100), (78,101),(208,183)
                 ,(259,171),(306,179),(341,183),(396,195),(395,245),(503,207),(555,210),(688,174),(762,207)
                 ,(76, 193),(157, 278),(243, 281),(395, 281),(502, 279),(555, 281),(395, 320),(480, 314),(529, 322)
                 ,(620, 289),(729, 262),(808, 239),(849, 231),(990, 226),(153, 346),(205, 345),(207, 389),(287, 350)
                 ,(342, 337),(489, 350),(394, 406),(491, 352),(558, 351),(544, 370),
                 (715, 348),(179, 445),(234, 443),(288, 444),(341, 443),(354, 440),(444, 443),(490, 405),(493, 457),
                 (621, 373),(691, 478),(960, 327),(181, 551),(109, 570),(605, 224),(396, 350),(73, 258),(840, 349)]

# Groups of three kinds of units
med_group = pygame.sprite.Group()
fire_group = pygame.sprite.Group()
joint_group = pygame.sprite.Group()

time_e, time_f = [], []
indicator = 0
uti_indicator = 0
draggingToken = False
tokenx, tokeny = None, None
move = False
COLOR = None
med_active = None
fire_active = None
joint_active = None


def asset_path(*parts):
    return os.path.join(BASE_DIR, *parts)


def make_gradient(start_rgb, end_rgb, steps):
    start = np.array(start_rgb, dtype=float)
    end = np.array(end_rgb, dtype=float)
    gradient = []
    for i in range(steps):
        ratio = i / max(steps - 1, 1)
        color = start + (end - start) * ratio
        gradient.append(tuple(int(round(x)) for x in color))
    return gradient


colors = make_gradient((244, 32, 8), (156, 245, 66), 20)

# Three classes
class Med_Unit(pygame.sprite.Sprite):
    def __init__(self, pos):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load(asset_path('figures', 'med.png'))
        self.image = pygame.transform.smoothscale(self.image, (ICONSIZE, ICONSIZE))
        self.rect = self.image.get_rect(center=pos)
        self.pos = pos

    def update(self,pos):
        self.rect = self.image.get_rect(center=pos)
        self.pos = pos

class Fire_Unit(pygame.sprite.Sprite):
    def __init__(self, pos):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load(asset_path('figures', 'fire.png'))
        self.image = pygame.transform.smoothscale(self.image, (ICONSIZE, ICONSIZE))
        self.rect = self.image.get_rect(center=pos)
        self.pos = pos

    def update(self,pos):
        self.rect = self.image.get_rect(center=pos)
        self.pos = pos

class Joint_Unit(pygame.sprite.Sprite):
    def __init__(self, pos):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load(asset_path('figures', 'joint.png'))
        self.image = pygame.transform.smoothscale(self.image, (ICONSIZE, ICONSIZE))
        self.rect = self.image.get_rect(center=pos)
        self.pos = pos

    def update(self, pos):
        self.rect = self.image.get_rect(center=pos)
        self.pos = pos

async def main():
    global FPSCLOCK, DISPLAYSURF, MEDPILERECT, FIREPILERECT, MEDTOKENIMG, JOINTTOKENIMG, JOINTPILERECT
    global FIRETOKENIMG, CALRECT, CALTOKENIMG, ORIBUTTONRECT, ORIBUTTONTOKENIMG, med_group, fire_group, joint_group
    global CLEARBUTTONRECT, CLEARBUTTONTOKENIMG, font, font_medium, font_large, INDIBUTTONRECT, INDIBUTTONTOKENIMG
    global TRACTS_IMAGE, census_latlong, UTIBUTTONRECT, UTIBUTTONTOKENIMG, BGMAP, BGTRACT

    census_latlong = census_tracts()

    pygame.init()
    font = pygame.font.Font('freesansbold.ttf', 14)
    font_medium = pygame.font.Font('freesansbold.ttf', 22)
    font_large = pygame.font.Font('freesansbold.ttf', 36)
    FPSCLOCK = pygame.time.Clock()
    DISPLAYSURF = pygame.display.set_mode((WINDOWWIDTH, WINDOWHEIGHT))
    pygame.display.set_caption('Saint Paul, MN')

    BGMAP = pygame.image.load(asset_path('figures', 'map.png'))
    BGMAP = pygame.transform.smoothscale(BGMAP, (WINDOWWIDTH, WINDOWHEIGHT))
    BGTRACT = pygame.image.load(asset_path('figures', 'tracts.png'))
    BGTRACT = pygame.transform.smoothscale(BGTRACT, (1073, 792))

    CALRECT = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), 0, ICONSIZE, ICONSIZE)

    MEDPILERECT = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(6 * ICONSIZE / 2), ICONSIZE, ICONSIZE)

    FIREPILERECT = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(3 * ICONSIZE / 2), ICONSIZE, ICONSIZE)

    JOINTPILERECT = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(9 * ICONSIZE / 2), ICONSIZE, ICONSIZE)

    ORIBUTTONRECT = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(12 * ICONSIZE / 2), ICONSIZE, ICONSIZE)

    CLEARBUTTONRECT = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(15 * ICONSIZE / 2), ICONSIZE, ICONSIZE)

    INDIBUTTONRECT = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(18 * ICONSIZE / 2), ICONSIZE, ICONSIZE)

    UTIBUTTONRECT = pygame.Rect(WINDOWWIDTH - int(3 * ICONSIZE / 2), int(21 * ICONSIZE / 2), ICONSIZE, ICONSIZE)


    MEDTOKENIMG = pygame.image.load(asset_path('figures', 'med.png'))
    MEDTOKENIMG = pygame.transform.smoothscale(MEDTOKENIMG, (ICONSIZE, ICONSIZE))
    FIRETOKENIMG = pygame.image.load(asset_path('figures', 'fire.png'))
    FIRETOKENIMG = pygame.transform.smoothscale(FIRETOKENIMG, (ICONSIZE, ICONSIZE))
    JOINTTOKENIMG = pygame.image.load(asset_path('figures', 'joint.png'))
    JOINTTOKENIMG = pygame.transform.smoothscale(JOINTTOKENIMG, (ICONSIZE, ICONSIZE))

    tracts_list = os.listdir(asset_path('tracts'))
    tracts_list.sort()

    TRACTS_IMAGE = []
    for image in tracts_list:
        if image.endswith('.png'):
            TRACT = pygame.image.load(asset_path('tracts', image))
            TRACT = pygame.transform.rotozoom(TRACT,0,0.58)
            TRACTS_IMAGE.append(TRACT)

    CALTOKENIMG = pygame.image.load(asset_path('figures', 'start.png'))
    CALTOKENIMG = pygame.transform.smoothscale(CALTOKENIMG, (ICONSIZE, ICONSIZE))
    
    ORIBUTTONTOKENIMG = pygame.image.load(asset_path('figures', 'stpaul.jpg'))
    ORIBUTTONTOKENIMG = pygame.transform.smoothscale(ORIBUTTONTOKENIMG, (ICONSIZE, ICONSIZE))

    CLEARBUTTONTOKENIMG = pygame.image.load(asset_path('figures', 'clear-icon.png'))
    CLEARBUTTONTOKENIMG = pygame.transform.smoothscale(CLEARBUTTONTOKENIMG, (ICONSIZE, ICONSIZE))

    INDIBUTTONTOKENIMG = pygame.image.load(asset_path('figures', 'switch.png'))
    INDIBUTTONTOKENIMG = pygame.transform.smoothscale(INDIBUTTONTOKENIMG, (ICONSIZE, ICONSIZE))

    UTIBUTTONTOKENIMG = pygame.image.load(asset_path('figures', 'uti.png'))
    UTIBUTTONTOKENIMG = pygame.transform.smoothscale(UTIBUTTONTOKENIMG, (ICONSIZE, ICONSIZE))
    running = True
    while running:
        running = getMove()
        await asyncio.sleep(0)

def fill(surface, COLOR):
    color = pygame.surfarray.pixels3d(surface)
    color[:, :] = COLOR
    del color

def drawBoard(extraToken=None):
    global census_latlong
    DISPLAYSURF.blit(BGMAP, (0,0))

    if len(time_e) > 0:
        if indicator == 0:
            for i in range(len(tracts_pix)):
                DISPLAYSURF.blit(TRACTS_IMAGE[i],tracts_pix[i])
            DISPLAYSURF.blit(BGTRACT, (70,7))
            for i in range(71):
                    text_e = font.render('{0:.2f}'.format(time_e[i]), True, BLACK) 
                    DISPLAYSURF.blit(text_e, corrdinates_to_pix(census_latlong[i]))
            text_mean_e = font_large.render('EMS: {0:.2f} min'.format(mean_e), True, RED) 
            DISPLAYSURF.blit(text_mean_e, (620,630))
            text_mean_f = font_medium.render('Fire: {0:.2f} min'.format(mean_f), True, BRIGHTBLUE)
            DISPLAYSURF.blit(text_mean_f, (620,670))
        elif indicator == 1:
            for i in range(len(tracts_pix)):
                #fill(TRACTS_IMAGE[i], colors[19-color_map_f[i]])
                DISPLAYSURF.blit(TRACTS_IMAGE[i],tracts_pix[i])
            DISPLAYSURF.blit(BGTRACT, (70,7))
            for i in range(71):
                    text_f = font.render('{0:.2f}'.format(time_f[i]), True, BLACK) 
                    DISPLAYSURF.blit(text_f, corrdinates_to_pix(census_latlong[i]))
            text_mean_e = font_medium.render('EMS: {0:.2f} min'.format(mean_e), True, RED) 
            DISPLAYSURF.blit(text_mean_e, (620,630))
            text_mean_f = font_large.render('Fire: {0:.2f} min'.format(mean_f), True, BRIGHTBLUE)
            DISPLAYSURF.blit(text_mean_f, (620,670))
        else:
            text_mean_e = font_medium.render('EMS: {0:.2f} min'.format(mean_e), True, RED) 
            DISPLAYSURF.blit(text_mean_e, (620,630))
            text_mean_f = font_medium.render('Fire: {0:.2f} min'.format(mean_f), True, BRIGHTBLUE)
            DISPLAYSURF.blit(text_mean_f, (620,670))

        if uti_indicator == 1:
            i_med = 0
            i_fire = 0
            for med in med_group:
                text_e_rho = font_medium.render('{0:.2f}%'.format(rho_e[i_med]*100), True, RED)
                pix = med.pos
                DISPLAYSURF.blit(text_e_rho, (pix[0]-30, pix[1]-35))
                i_med += 1
            for fire in fire_group:
                text_f_rho = font_medium.render('{0:.2f}%'.format(rho_f[i_fire]*100), True, BRIGHTBLUE) 
                pix = fire.pos
                DISPLAYSURF.blit(text_f_rho, (pix[0]-25, pix[1]+15))
                i_fire += 1
            for joint in joint_group:
                text_e_rho = font_medium.render('{0:.2f}%'.format(rho_e[i_med]*100), True, RED)
                text_f_rho = font_medium.render('{0:.2f}%'.format(rho_f[i_fire]*100), True, BRIGHTBLUE)
                pix = joint.pos
                DISPLAYSURF.blit(text_e_rho, (pix[0]-25, pix[1]-35))
                DISPLAYSURF.blit(text_f_rho, (pix[0]-25, pix[1]+15))
                i_med += 1
                i_fire += 1

    med_group.draw(DISPLAYSURF)
    fire_group.draw(DISPLAYSURF)
    joint_group.draw(DISPLAYSURF)

    DISPLAYSURF.blit(MEDTOKENIMG, MEDPILERECT)
    DISPLAYSURF.blit(FIRETOKENIMG, FIREPILERECT)
    DISPLAYSURF.blit(JOINTTOKENIMG, JOINTPILERECT)
    DISPLAYSURF.blit(CALTOKENIMG, CALRECT)
    DISPLAYSURF.blit(ORIBUTTONTOKENIMG, ORIBUTTONRECT)
    DISPLAYSURF.blit(CLEARBUTTONTOKENIMG, CLEARBUTTONRECT)
    DISPLAYSURF.blit(INDIBUTTONTOKENIMG, INDIBUTTONRECT)
    DISPLAYSURF.blit(UTIBUTTONTOKENIMG, UTIBUTTONRECT)

    if extraToken is not None:
        if extraToken['color'] == MED:
            DISPLAYSURF.blit(MEDTOKENIMG, (extraToken['x'], extraToken['y'], ICONSIZE, ICONSIZE))
        elif extraToken['color'] == FIRE:
            DISPLAYSURF.blit(FIRETOKENIMG, (extraToken['x'], extraToken['y'], ICONSIZE, ICONSIZE))
        elif extraToken['color'] == JOINT:
            DISPLAYSURF.blit(JOINTTOKENIMG, (extraToken['x'], extraToken['y'], ICONSIZE, ICONSIZE))

def getMove():
    global time_e, time_f, indicator, uti_indicator, draggingToken, tokenx, tokeny, move, COLOR
    global med_active, fire_active, joint_active

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            return False
        elif event.type == MOUSEBUTTONDOWN and not draggingToken and MEDPILERECT.collidepoint(event.pos):
            draggingToken = True
            tokenx, tokeny = event.pos
            COLOR = MED
        elif event.type == MOUSEBUTTONDOWN and not draggingToken and FIREPILERECT.collidepoint(event.pos):
            draggingToken = True
            tokenx, tokeny = event.pos
            COLOR = FIRE
        elif event.type == MOUSEBUTTONDOWN and not draggingToken and JOINTPILERECT.collidepoint(event.pos):
            draggingToken = True
            tokenx, tokeny = event.pos
            COLOR = JOINT
        elif event.type == MOUSEMOTION and draggingToken:
            if move:
                if COLOR == MED:
                    med_active.update(event.pos)
                elif COLOR == FIRE:
                    fire_active.update(event.pos)
                elif COLOR == JOINT:
                    joint_active.update(event.pos)
            tokenx, tokeny = event.pos
        elif event.type == MOUSEBUTTONUP and draggingToken:
            if not move:
                if COLOR == MED:
                    med_group.add(Med_Unit(event.pos))
                elif COLOR == FIRE:
                    fire_group.add(Fire_Unit(event.pos))
                elif COLOR == JOINT:
                    joint_group.add(Joint_Unit(event.pos))
            tokenx, tokeny = None, None
            draggingToken = False
            move = False
            if (len(med_group) + len(joint_group)) >= 2 and (len(fire_group) + len(joint_group)) >= 2:
                analyze()
        elif event.type == MOUSEBUTTONDOWN and not draggingToken and CALRECT.collidepoint(event.pos):
            if (len(med_group) + len(joint_group)) >= 2 and (len(fire_group) + len(joint_group)) >= 2:
                analyze()
        elif event.type == MOUSEBUTTONDOWN and not draggingToken and ORIBUTTONRECT.collidepoint(event.pos):
            station_latlong = saint_paul_stations()
            for i, station in enumerate(station_latlong):
                if i < 3:
                    pix = corrdinates_to_pix(station)
                    med_group.add(Med_Unit((pix[0] - 20, pix[1])))
                elif i < 8:
                    fire_group.add(Fire_Unit(corrdinates_to_pix(station)))
                else:
                    joint_group.add(Joint_Unit(corrdinates_to_pix(station)))
        elif event.type == MOUSEBUTTONDOWN and event.button == RIGHT:
            for group in (joint_group, fire_group, med_group):
                for unit in list(group):
                    if unit.rect.collidepoint(event.pos):
                        unit.kill()
                        analyze()
            if INDIBUTTONRECT.collidepoint(event.pos):
                indicator = 2
        elif event.type == MOUSEBUTTONDOWN and not draggingToken and CLEARBUTTONRECT.collidepoint(event.pos):
            med_group.empty()
            fire_group.empty()
            joint_group.empty()
            time_e, time_f = [], []
        elif event.type == MOUSEBUTTONDOWN and not draggingToken and INDIBUTTONRECT.collidepoint(event.pos):
            if event.button == LEFT:
                indicator = (indicator + 1) % 2
                if len(time_e) > 0:
                    active_map = color_map_e if indicator == 0 else color_map_f
                    for i in range(len(tracts_pix)):
                        fill(TRACTS_IMAGE[i], colors[19 - active_map[i]])
        elif event.type == MOUSEBUTTONDOWN and not draggingToken and UTIBUTTONRECT.collidepoint(event.pos):
            uti_indicator = (uti_indicator + 1) % 2
        elif event.type == MOUSEBUTTONDOWN:
            for group, unit_name in ((joint_group, JOINT), (fire_group, FIRE), (med_group, MED)):
                for unit in list(group):
                    if unit.rect.collidepoint(event.pos):
                        tokenx, tokeny = event.pos
                        unit.update(event.pos)
                        draggingToken = True
                        move = True
                        COLOR = unit_name
                        if unit_name == JOINT:
                            joint_active = unit
                        elif unit_name == FIRE:
                            fire_active = unit
                        else:
                            med_active = unit

    if tokenx is not None and tokeny is not None:
        drawBoard({'x': tokenx - int(ICONSIZE / 2), 'y': tokeny - int(ICONSIZE / 2), 'color': COLOR})
    else:
        drawBoard()

    pygame.display.update()
    FPSCLOCK.tick(FPS)
    return True
def analyze():
    global time_e, time_f, mean_e, mean_f, rho_e, rho_f, color_map_e, color_map_f
    latlong_list = []
    for med in med_group:
        latlong_list += [pix_to_corrdinates(med.pos)]
    for fire in fire_group:
        latlong_list += [pix_to_corrdinates(fire.pos)]
    for joint in joint_group:
        latlong_list += [pix_to_corrdinates(joint.pos)]

    N_1, N_2, N_D = len(med_group), len(fire_group), len(joint_group)
    distance_matrix = get_distance(latlong_list)*1.37

    distance_matrix = distance_matrix/1600
    distance_matrix[distance_matrix > 0.7712735431536174] = (distance_matrix[distance_matrix > 0.7712735431536174]*111.51113889304331+86.005591195132666)/60
    distance_matrix[distance_matrix < 0.7712735431536174] = 195.86302790816589*np.sqrt(distance_matrix[distance_matrix < 0.7712735431536174])/60

    N = N_1+N_2+N_D
    units_e = np.sort([x for x in range(N) if x not in list(range(N_1,N_1+N_2))])
    units_f = np.sort([x for x in range(N) if x not in list(range(N_1))])

    t_mat_1 = distance_matrix[:, units_e]
    t_mat_2 = distance_matrix[:, units_f]
    pre_list_1 = t_mat_1.argsort(axis=1)
    pre_list_2 = t_mat_2.argsort(axis=1)

    Data = {}
    Data['N'], Data['N_1'], Data['N_2'], Data['K'] = N, N_1, N_2, 71
    Data['Mu_1'], Data['Mu_2'] = 42.17, 51.3
    Data['Lambda_1'], Data['Lambda_2'] = 57.80867911518289, 47.266999999999996 - 7
    Data['Pre_L_1'], Data['Pre_L_2'] = pre_list_1, pre_list_2
    Data['e'], Data['f'] = np.array([ 0.01158147,  0.02600284,  0.00940717,  0.0192137 ,  0.01717252,
         0.01606319,  0.02067803,  0.01757188,  0.01091587,  0.01672879,
         0.00603479,  0.00891906,  0.00993965,  0.0115371 ,  0.02116613,
         0.01078275,  0.03572062,  0.01792687,  0.01797125,  0.00559105,
         0.01388889,  0.00301739,  0.00563543,  0.00745474,  0.01610756,
         0.01238019,  0.00954029,  0.03310259,  0.00554668,  0.01766063,
         0.00261803,  0.02671282,  0.01690628,  0.00474796,  0.02334043,
         0.00519169,  0.00505857,  0.01393326,  0.03421193,  0.00803159,
         0.01167022,  0.03625311,  0.03518814,  0.00532481,  0.00306177,
         0.0041711 ,  0.00430422,  0.00430422,  0.0056798 ,  0.00230742,
         0.00736599,  0.01215832,  0.00736599,  0.0063454 ,  0.00789847,
         0.00616791,  0.00412673,  0.01042776,  0.02236422,  0.00940717,
         0.00954029,  0.00430422,  0.03430067,  0.00745474,  0.02538161,
         0.01264643,  0.03421193,  0.06948882,  0.00794285,  0.01206958,
         0.00075435]), np.array([ 0.0144893 ,  0.02629539,  0.00888438,  0.01252161,  0.01222348,
         0.01329676,  0.03631268,  0.01103095,  0.00840737,  0.01532407,
         0.00924214,  0.00590305,  0.01234273,  0.00775148,  0.00727446,
         0.01043468,  0.04239461,  0.02593763,  0.02522211,  0.00673782,
         0.01168684,  0.01109057,  0.00453163,  0.00816886,  0.01043468,
         0.00787073,  0.00643969,  0.01454892,  0.00888438,  0.0267724 ,
         0.00465088,  0.02426808,  0.01639735,  0.00590305,  0.01341602,
         0.00608193,  0.00787073,  0.01699362,  0.094568  ,  0.00930177,
         0.01109057,  0.02581838,  0.02504323,  0.00697633,  0.0035776 ,
         0.0051279 ,  0.00524715,  0.00572417,  0.00918252,  0.00381611,
         0.0111502 ,  0.01454892,  0.00798998,  0.00942102,  0.00477014,
         0.00381611,  0.0051279 ,  0.00679745,  0.01603959,  0.00709558,
         0.06159442,  0.00548566,  0.02098861,  0.00965953,  0.0247451 ,
         0.00954028,  0.02045197,  0.01168684,  0.01753026,  0.00673782,
         0.00131179])

    K = Data['K']
    Lambda_1, Lambda_2 = Data['Lambda_1'], Data['Lambda_2']
    Mu_1, Mu_2 = Data['Mu_1'], Data['Mu_2']
    frac_j_1, frac_j_2 = Data['e'], Data['f']
    
    SP_joint = dm.Three_State_Hypercube({'Lambda_1':Lambda_1, 'Mu_1': Mu_1, 'Lambda_2':Lambda_2, 'Mu_2': Mu_2})
    SP_joint.Update_Parameters(N = N, N_1 = N_1, N_2 = N_2, K = K, pre_list_1 = pre_list_1, pre_list_2 = pre_list_2 , frac_j_1 = frac_j_1, frac_j_2 = frac_j_2, t_mat_1=t_mat_1, t_mat_2=t_mat_2)
    SP_joint.Creat_Two_Subsystems()
    SP_joint.Linear_Alpha()
    rho_e, rho_f = SP_joint.sub1.rho_approx, SP_joint.sub2.rho_approx
    MRT_1, MRT_2, MRT_1_j, MRT_2_j = SP_joint.Get_MRT_Approx_3state()
    time_e = MRT_1_j + 1.75
    time_f = MRT_2_j + 1.67 - 0.04
    mean_e = MRT_1 + 1.75
    mean_f = MRT_2 + 1.67 - 0.04

    color_map_e = np.array([int((i-4)*20/6) for i in time_e])
    color_map_e[color_map_e<0] = 0
    color_map_e[color_map_e>19] = 19

    color_map_f = np.array([int((i-4)*20/6) for i in time_f])
    color_map_f[color_map_f<0] = 0
    color_map_f[color_map_f>19] = 19
    if indicator == 0:
        for i in range(len(tracts_pix)):
            fill(TRACTS_IMAGE[i], colors[19-color_map_e[i]])
    elif indicator == 1:
        for i in range(len(tracts_pix)):
            fill(TRACTS_IMAGE[i], colors[19-color_map_f[i]])

def get_distance(latlong_list):
    latlong_census = census_tracts()
    return np.array([[haversine_meters(i, j) for j in latlong_list] for i in latlong_census], dtype=float)


def haversine_meters(point_a, point_b):
    lat1, lon1 = np.radians(point_a)
    lat2, lon2 = np.radians(point_b)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def pix_to_corrdinates(pos):
    Lat = UPLEFT[0]+ (pos[1]-UPLEFT_PIX[1])*(DOWNRIGHT[0]-UPLEFT[0])/(DOWNRIGHT_PIX[1]-UPLEFT_PIX[1])
    Long = UPLEFT[1]+ (pos[0]-UPLEFT_PIX[0])*(DOWNRIGHT[1]-UPLEFT[1])/(DOWNRIGHT_PIX[0]-UPLEFT_PIX[0])
    return (Lat, Long)

def corrdinates_to_pix(latlong):
    pos1 = UPLEFT_PIX[1]+ (latlong[0]-UPLEFT[0])*(DOWNRIGHT_PIX[1]-UPLEFT_PIX[1])/(DOWNRIGHT[0]-UPLEFT[0])
    pos2 = UPLEFT_PIX[0]+ (latlong[1]-UPLEFT[1])*(DOWNRIGHT_PIX[0]-UPLEFT_PIX[0])/(DOWNRIGHT[1]-UPLEFT[1])
    return (pos2, pos1)

def census_tracts():
    latlong = [(44.97985,-93.1929),
                (44.97791,-93.15685),
                (44.98301,-93.13659),
                (44.98617,-93.11737),
                (44.98301,-93.0978),
                (44.98714,-93.06999),
                (44.98471,-93.02523),
                (44.97524,-93.11702),
                (44.97451,-93.08303),
                (44.97597,-93.07205),
                (44.97912,-93.05522),
                (44.97062,-93.13556),
                (44.96892,-93.11599),
                (44.96941,-93.0978),
                (44.96771,-93.07754),
                (44.97184,-93.05625),
                (44.96917,-93.04149),
                (44.97029,-93.02022),
                (44.96771,-93.19221),
                (44.96286,-93.17722),
                (44.96406,-93.16915),
                (44.96115,-93.16228),
                (44.96285,-93.15199),
                (44.96334,-93.13728),
                (44.95823,-93.13693),
                (44.96091,-93.12114),
                (44.96018,-93.11153),
                (44.95993,-93.08372),
                (44.96188,-93.06896),
                (44.95872,-93.19255),
                (44.95022,-93.18431),
                (44.95143,-93.16097),
                (44.95362,-93.13728),
                (44.95337,-93.12114),
                (44.95337,-93.11153),
                (44.94876,-93.13934),
                (44.94876,-93.12629),
                (44.94827,-93.11325),
                (44.94852,-93.09196),
                (44.95167,-93.07205),
                (44.95532,-93.06003),
                (44.95409,-93.04558),
                (44.95676,-93.02021),
                (44.94098,-93.18775),
                (44.9439,-93.17539),
                (44.93782,-93.17539),
                (44.94001,-93.16234),
                (44.94098,-93.15273),
                (44.94603,-93.12115),
                (44.93636,-93.13693),
                (44.94074,-93.12286),
                (44.94258,-93.11019),
                (44.93532,-93.11242),
                (44.93515,-93.06827),
                (44.92616,-93.18328),
                (44.92713,-93.17264),
                (44.92689,-93.16268),
                (44.92397,-93.15204),
                (44.91838,-93.14071),
                (44.92591,-93.13282),
                (44.93053,-93.1232),
                (44.92421,-93.1081),
                (44.93564,-93.09333),
                (44.9247,-93.07857),
                (44.93418,-93.01574),
                (44.91254,-93.17024),
                (44.89782,-93.17799),
                (44.95799,-93.09883),
                (44.94293,-93.13797),
                (44.93831,-93.19733),
                (44.91862,-93.03428)]
    return latlong

def saint_paul_stations():
    stations_latlong = [(44.95099318068965, -93.08500943661971), (44.98105367172413, -93.03405129577465), (44.97555152827586, -93.18043666666667), (44.94884600275862, -93.13195963380282), (44.97179396689655, -93.0447391455399), (44.95099318068965, -93.08500943661971), (44.98091947310345, -93.03386044131456), (44.97555152827586, -93.18043666666667), (44.93005819586207, -93.13863953990611), (44.965620830344825, -93.08577285446009), (44.92911880551724, -93.09207105164319), (44.95220096827586, -93.16650429107982), (44.98293245241379, -93.06172519248827), (44.962936857931034, -93.13195963380282), (44.91717512827586, -93.1760470140845), (44.97420954206896, -93.10905709859155), (44.95179837241379, -93.02794395305165)]
    return stations_latlong

if __name__ == '__main__':
    asyncio.run(main())
