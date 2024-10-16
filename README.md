# prim_diffusion
Telecom Paris Project

Objectives:
The objectives and perspectives for this project are as follows:
1. Use diffusion models [1] for generation of binarized images (multiclass masks) of cardiac shapes
2. Implement the blurring diffusion models of [2]
3. Combine blurred diffusion models with the VAE framework


## Dataset

### ACDC
ED: End-Diastole（舒张末期）
ES: End-Systole（收缩末期）

**A (Anterior)**：前方，表示病人的前面（朝向病人的脸部）。

**P (Posterior)**：后方，表示病人的背面（朝向病人的背部）。

**L (Left)**：左方，表示病人的左侧。

**R (Right)**：右方，表示病人的右侧。

**label**

| value | class                                 |
| ----- | ------------------------------------- |
| 0     | background                            |
| 1     | Right Ventricle, RV                   |
| 2     | Left Ventricle, LV                    |
| 3     | Myocardium of the Left Ventricle, MYO |

![img](https://www.myheart.org.sg/wp-content/uploads/2022/01/How-Your-Heart-Works-2-scaled.jpg)

**Class**
- NOR: 
    - Examination with normal cardiac anatomy and function
- MINF: 
    - Patients with a systolic heart failure with infarction
- DCM: 
    - Patients with dilated cardiomyopathy have an ejection fraction below 40%, a LV volume greater than 100 mL/m2 and a wall thickness in diastole smaller than 12 mm.
- HCM:
    - Patients with hypertrophic cardiomyopathy, i.e. a normal cardiac function (ejection fraction greater than 55%) but with myocardial segments thicker than 15 mm in diastole.
- ARV
    - : Patients with abnormal right ventricle have a RV volume greater than 110 mL/m2 for men, and greater than 100 mL/m2 for women , or/and a RV ejection fraction below 40%. 



| 中文名称 | 英文名称         | 缩写 |
| -------- | ---------------- | ---- |
| 右心房   | Right Atrium     | RA   |
| 右心室   | Right Ventricle  | RV   |
| 左心房   | Left Atrium      | LA   |
| 左心室   | Left Ventricle   | LV   |
| 心肌     | Myocardium       | -    |
| 二尖瓣   | Mitral Valve     | MV   |
| 三尖瓣   | Tricuspid Valve  | TV   |
| 主动脉   | Aorta            | -    |
| 肺动脉   | Pulmonary Artery | PA   |
| 肺静脉   | Pulmonary Vein   | PV   |
| 冠状动脉 | Coronary Artery  | CA   |
| 心包     | Pericardium      | -    |
| 心房     | Atria            | -    |
| 心室     | Ventricles       | -    |



## Basic cardiac views

- Long Axis VLA, RAO
- Short axis SA, SAX
- Four chamber



## Appendix

- [心脏磁共振基本定位](https://www.bilibili.com/video/BV14f4y1j7ea/?vd_source=cf7bd5044042040cb9df7f8c42c5d791)
- 





























