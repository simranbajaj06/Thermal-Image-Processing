# Research papers and datasets referenced


## 3. Research, papers and datasets referenced


### Provided with the assignment

- Journal of Thermal Biology (2023), article PII S0306-4565(23)00005-0: https://www.sciencedirect.com/science/article/abs/pii/S0306456523000050 [add authors, title and year from the article page]
- Thermal Infrared Imaging to Evaluate Emotional Competences in Nursing Students: A First Approach through a Case Study. Sensors (2020), PMC7248891: https://pmc.ncbi.nlm.nih.gov/articles/PMC7248891/ [add authors from the article page]


### Physiological basis of skin temperature and thermography

- Pennes HH. Analysis of tissue and arterial blood temperatures in the resting human forearm. J Appl Physiol 1948;1(2):93-122.
- Ring EFJ, Ammer K. Infrared thermal imaging in medicine. Physiological Measurement 2012;33(3):R33-R46.
- Garbey M, Sun N, Merla A, Pavlidis I. Contact-free measurement of cardiac pulse based on the analysis of thermal imagery. IEEE Trans Biomed Eng 2007;54(8):1418-1426.


### Dynamic thermography after a cold challenge (related literature found by search - verify bibliographic details before citing)

- Variability in peripheral rewarming after cold stress among 255 healthy Norwegian army conscripts assessed by dynamic infrared thermography. Int J Circumpolar Health (2018). https://www.tandfonline.com/doi/full/10.1080/22423982.2018.1536250
- Dynamic infrared imaging for analysis of fingertip temperature after cold water stimulation and neurothermal modeling study. Comput Biol Med (2010). https://www.sciencedirect.com/science/article/abs/pii/S0010482510000806
- Cold stress dynamic thermography for evaluation of vascular disorders in hand-arm vibration syndrome (title as found in search results).
- Dynamic infrared thermography reveals bilateral thermal recovery asymmetry in scoliosis. PMC13413119 (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC13413119/).


### Image-analysis and statistics methods used

- Otsu N. A threshold selection method from gray-level histograms. IEEE Trans Syst Man Cybern 1979;9(1):62-66.
- Savitzky A, Golay MJE. Smoothing and differentiation of data by simplified least squares procedures. Anal Chem 1964;36(8):1627-1639.
- Lucas BD, Kanade T. An iterative image registration technique with an application to stereo vision. Proc IJCAI 1981, pp. 674-679.
- Horn BKP, Schunck BG. Determining optical flow. Artificial Intelligence 1981;17:185-203.
- Evangelidis GD, Psarakis EZ. Parametric image alignment using enhanced correlation coefficient maximization. IEEE Trans Pattern Anal Mach Intell 2008;30(10):1858-1865.
- Mardia KV, Jupp PE. Directional Statistics. Wiley, 2000 (circular mean and mean resultant length R).


### Dataset used

- DIRT-SR38: Real Paired Cross-Sensor Thermal Sequence Dataset. Zenodo, DOI 10.5281/zenodo.19241293 (CC BY-NC 4.0). 38 subjects, 30,780 aligned low-/high-resolution thermal image pairs, standardized abdominal cooling-rewarming protocol, 8-bit and 16-bit PNG, relative (not absolute-temperature) intensities. Companion paper: Clarys et al., CVPR Workshops 2026 (cited as requested on the dataset page).

The two references supplied with the assignment concern thermography of physiological states; they motivate the use of surface temperature as a physiological signal but do not describe a propagation-direction algorithm, which is why the method here is built from the physical model and standard image-analysis tools listed above.


### Dataset used: DIRT-SR38

DIRT-SR38 contains abdominal thermal image sequences of 38 subjects recorded under a standardised cooling-rewarming protocol, stored as aligned low- and high-resolution pairs (it was published for thermal super-resolution). In the folder layout used here (dataset_export/8bit/<subject>/HR|LR/dynamic|static) the 'dynamic' folders hold the time sequences analysed in this report (high-resolution versions). Values are relative intensities, not degrees Celsius.
